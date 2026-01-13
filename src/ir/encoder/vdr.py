from contextlib import nullcontext
from functools import partial
import logging
from typing import List, Union

import torch
from torch import Tensor as T
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertConfig, BatchEncoding, PreTrainedModel

from ..utils.sparsify_utils import build_bow_mask, build_topk_mask, elu1p
from ..utils.visualize_utils import wordcloud_from_dict
from ..training.ddp_utils import get_rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class VDREncoderConfig(BertConfig):
    """
    VDR编码器配置类，继承自BERT配置类
    
    VDR (Vocabulary Disentangled Retrieval) 的核心思想是将文本表示为词汇空间中的稀疏向量，
    每个维度对应一个词汇表中的token，实现可解释的语义检索。

    Args:
        model_id (str): 基础模型标识符。默认为 'bert-base-uncased'，使用BERT作为骨干网络
        max_len (int): 输入序列的最大长度。默认为 256
        norm (bool): 是否对输出向量进行归一化。默认为 False
        shift_vocab_num (int): 词汇表偏移量，用于跳过特殊token（如[PAD], [UNK], [CLS]等）。
                               对于bert-base-uncased，默认999可以跳过前999个特殊符号和标点
    """

    def __init__(
        self,
        model_id='bert-base-uncased',
        max_len=256,
        norm=False,
        shift_vocab_num=999,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.model_id = model_id
        self.norm = norm
        self.shift_vocab_num = shift_vocab_num


class VDREncoder(PreTrainedModel):
    """
    VDR编码器：将文本映射到词汇空间的稀疏向量表示
    
    核心创新：
    1. 使用BERT的词嵌入矩阵作为"码本"(Codebook)
    2. 将文本的上下文表示投影到词汇空间，每个维度代表一个token的激活强度
    3. 通过稀疏化实现可解释性：只保留最相关的token维度
    """
    config_class = VDREncoderConfig

    def __init__(self, config: VDREncoderConfig, **kwargs):
        """初始化VDR编码器
        
        关键组件：
        - bert_model: 用于提取上下文感知的token表示
        - ln: Layer Normalization，稳定训练
        - build_bow_mask: 构建词袋(Bag-of-Words)掩码的函数，用于标识输入中出现的token
        """
        super().__init__(config, **kwargs)
        self.config = config
        # Layer Normalization用于稳定特征分布
        self.ln = torch.nn.LayerNorm(self.config.hidden_size)
        # 加载预训练BERT模型（不使用pooling层，我们需要所有token的表示）
        self.bert_model = AutoModel.from_pretrained(config.model_id, add_pooling_layer=False)
        # 加载对应的tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        # 创建构建BOW掩码的偏函数，预先绑定词汇表大小和偏移量参数
        self.build_bow_mask = partial(build_bow_mask, vocab_size=config.vocab_size, shift_num=config.shift_vocab_num, norm=config.norm)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
    ) -> T:
        """
        前向传播：将输入文本转换为词汇空间的表示
        
        算法流程（VDR的核心）：
        1. 通过BERT获取上下文化的token表示 [batch, seq_len, hidden_dim]
        2. Layer Normalization归一化
        3. 与词嵌入矩阵相乘，投影到词汇空间 [batch, seq_len, vocab_size]
           - 这里的关键思想：使用BERT的词嵌入权重作为"码本"
           - 计算每个位置的表示与所有词汇的相似度
        4. 应用elu1p激活函数（ELU(x)+1，保证非负）
        5. 跨序列池化（max或mean），得到 [batch, vocab_size]
        6. 可选的L2归一化
        
        Args:
            input_ids: token ID序列 [batch_size, seq_len]
            token_type_ids: segment ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            vocab_emb: 词汇空间的表示向量 [batch_size, vocab_size]
        """
        # Step 1: BERT编码，获取上下文表示
        outputs = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Step 2: Layer Normalization
        last_hidden_state_ln = self.ln(last_hidden_state)
        
        # Step 3: 投影到词汇空间
        # 使用词嵌入矩阵的转置作为投影矩阵
        # shift_vocab_num: 跳过特殊token（[PAD], [UNK], [CLS], [SEP]等）
        vocab_embs = last_hidden_state_ln @ self.bert_model.embeddings.word_embeddings.weight[self.config.shift_vocab_num:, :].t()
        # [batch, seq_len, hidden_dim] @ [hidden_dim, vocab_size] = [batch, seq_len, vocab_size]
        
        # Step 4: 激活函数（ELU + 1，保证输出非负，类似ReLU但更平滑）
        vocab_embs = elu1p(vocab_embs)
        
        # Step 5: 序列维度池化
        if self.config.pooling == "max":
            # Max pooling: 取每个词汇维度在序列上的最大激活值
            vocab_emb = vocab_embs.max(1)[0]  # [batch, vocab_size]
        elif self.config.pooling == "mean":
            # Mean pooling: 可选择先top-k再平均
            if self.config.pooling_topk:
                vocab_emb = vocab_embs.topk(self.config.pooling_topk, dim=1).values.mean(1)
            else:
                vocab_emb = vocab_emb.mean(1)
        else:
            raise NotImplementedError
        
        # Step 6: 可选的L2归一化
        vocab_emb = F.normalize(vocab_emb) if self.config.norm else vocab_emb
        return vocab_emb

    def encode(
        self,
        texts: Union[List[str], str], 
        max_len: int = None, 
    ) -> BatchEncoding:
        max_len = max_len or self.config.max_len
        texts = [texts] if isinstance(texts, str) else texts
        encoding = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        encoding = encoding.to(self.device)
        return encoding

    def embed(
        self, 
        texts: Union[List[str], str], 
        batch_size: int = 128, 
        max_len: int = None, 
        topk: int = None,
        bow: bool = False, 
        activate_lexical: bool = True,
        require_grad: bool = False,
        to_cpu: bool = False,
        convert_to_tensor: bool = True,
        show_progress_bar: bool = False,
        **kwargs
    ) -> T:
        """将文本嵌入为词汇空间的稀疏表示
        
        这是VDR的核心推理接口，实现了三种表示模式：
        1. 纯词袋(BOW)表示：bow=True，仅标记出现的词汇
        2. 稠密表示：topk=-1，保留所有维度
        3. 稀疏表示：topk=K，仅保留top-K个最重要的维度（VDR的标准模式）

        Args:
            texts (str, List[str]): 待嵌入的文本或文本列表
            batch_size (int): 批处理大小。默认128
            max_len (int): 最大序列长度
            topk (int): Top-K稀疏化后的活跃维度数量
                - topk=0: 仅激活输入文本中出现的token对应的维度（词袋模式）
                - topk=-1 or None: 激活所有维度（稠密模式）
                - topk=K: 仅激活top-K个维度（稀疏模式，VDR标准设置）
            bow (bool): 如果为True，直接返回二值词袋表示（忽略神经网络计算）
            activate_lexical (bool): 是否强制激活输入token对应的词汇维度
                - True: 保证输入的token一定被激活（BOW mask ∪ Top-K mask）
                - False: 只使用Top-K mask
            require_grad (bool): 是否保留梯度用于反向传播
                - True: 保持训练模式，保留梯度
                - False: 切换到评估模式，确保推理一致性
            to_cpu (bool): 是否将结果移至CPU内存
            convert_to_tensor (bool): 是否返回Tensor（False则返回NumPy数组）
            show_progress_bar (bool): 是否显示进度条

        Returns:
            Tensor: 词汇空间的表示，形状为 [N, V]
                N: 文本数量
                V: 词汇表大小（通常为30522 - 999 = 29523 for BERT）
                
        工作原理：
            对于输入"Einstein developed the theory of relativity":
            1. BERT编码 → 上下文表示
            2. 投影到词汇空间 → 30K维向量（每个维度对应一个词）
            3. Top-K稀疏化 → 保留768个最相关的词
            4. 最终向量中，"einstein"、"theory"、"relativity"等维度值高
        """

        # 参数准备
        max_len = max_len or self.config.max_len
        topk = topk or self.config.topk
        texts = [texts] if isinstance(texts, str) else texts
        is_training = self.training

        # 如果不需要梯度且模型在训练模式，切换到评估模式以保证一致性
        if not require_grad and is_training:
            self.eval()

        with torch.no_grad() if not require_grad else nullcontext():
            batch_embs = []
            num_text = len(texts)
            iterator = range(0, num_text, batch_size)
            # 批次处理文本
            for batch_start in tqdm(iterator) if show_progress_bar else iterator:
                
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: batch_start : {batch_start}")
                batch_texts = texts[batch_start : batch_start + batch_size]
                
                # Step 1: Tokenization - 将文本转换为token IDs
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: encode_start : {batch_start}")
                encoding = self.encode(batch_texts, max_len=max_len)
                
                # Step 2: 构建BOW mask - 标记输入中出现过的token
                # BOW mask是一个二值矩阵，对应输入token的位置为1，其他为0
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: build_bow_start : {batch_start}")
                bow_mask = self.build_bow_mask(encoding.input_ids)
                
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: forward_start : {batch_start}")

                if bow:
                    # 模式1: 纯BOW表示，直接返回二值掩码
                    batch_emb = bow_mask
                else:
                    # 模式2/3: 稀疏或稀疏表示
                    # Step 3: 前向传播获取词汇空间表示
                    batch_emb = self(**encoding)
                    
                    # Step 4: 构建Top-K mask
                    if topk == 0: 
                        # 模式2a: 仅激活输入token对应的维度（与bow等价）
                        topk_mask = torch.zeros_like(batch_emb)
                    elif topk == None or topk == -1: 
                        # 模式2b: 激活所有维度（稀密表示）
                        topk_mask = torch.ones_like(batch_emb)
                    else: 
                        # 模式3: 激活top-k个最大的维度（VDR标准模式）
                        # 这是VDR的核心：通过稀疏化实现可解释性
                        topk_mask = build_topk_mask(batch_emb, topk)
                    
                    # Step 5: 组合BOW mask和Top-K mask
                    # activate_lexical=True: 保证输入token一定被激活（并集）
                    # activate_lexical=False: 只使用Top-K mask
                    mask = torch.logical_or(bow_mask, topk_mask) if activate_lexical else topk_mask
                    # 应用mask，将非激活维度置0
                    batch_emb *= mask
                    
                batch_embs.append(batch_emb)
            
            # 合并所有batch的结果
            emb = torch.cat(batch_embs, dim=0)
            
            # 格式转换
            if not convert_to_tensor:
                emb = emb.cpu().numpy()
            elif to_cpu:
                emb = emb.cpu()

        # 恢复原来的训练状态
        if is_training and not self.training:
            self.train()
        return emb

    def disentangle(self, text: str, topk: int = 768, visual=False, save_file=None):
        """
        解缠文本表示：将文本分解为最相关的词汇及其权重
        
        这是VDR的核心功能之一：提供可解释性。通过查看最高激活的词汇维度，
        我们可以理解模型认为哪些词汇对文本最重要。
        
        例如，对于输入"Albert Einstein developed the theory of relativity"：
        - 可能得到：{"einstein": 15.3, "relativity": 12.8, "theory": 10.5, "physics": 8.2, ...}
        - 这显示了模型捕捉到的语义概念，包括输入中未出现的相关词（如"physics"）
        
        Args:
            text (str): 待分析的文本
            topk (int): 返回前topk个最相关的词汇。默认768
            visual (bool): 是否生成词云图可视化
            save_file (str): 词云图保存路径
            
        Returns:
            dict: 字典，键为token，值为激活强度
                  按激活强度降序排列
        
        应用场景：
        1. 模型解释：理解模型关注什么
        2. 调试分析：检查模型是否捕捉到正确的语义
        3. 知识发现：发现文本的隐含语义关联
        """
        # Step 1: 获取文本的词汇空间表示，并取top-k
        topk_result = self.embed(text).topk(topk)
        topk_token_ids = topk_result.indices.flatten().tolist()
        
        # Step 2: 过滤掉shift之前的特殊token，还原为实际的token ID
        topk_token_ids = [x + self.config.shift_vocab_num for x in topk_token_ids if x >= self.config.shift_vocab_num]
        topk_values = topk_result.values.flatten().tolist()
        
        # Step 3: 将token ID转换为可读的token字符串
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_token_ids)
        
        # Step 4: 构建token→权重的字典
        results = dict(zip(topk_tokens, topk_values))
        
        # Step 5: 可选：生成词云图可视化
        if visual:
            wordcloud_from_dict(results, max_words=topk, save_file=save_file)
        
        return results

    # 别名，简写为dst
    dst = disentangle

