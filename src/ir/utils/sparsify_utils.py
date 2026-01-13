import torch
import torch.nn.functional as F
from typing import Tuple, List, Union
import numpy as np

# ELU激活函数 + 1，保证输出非负
# ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
# ELU(x) + 1 在x=0时输出1，在x<0时输出接近0，在x>0时输出>1
# 相比ReLU更平滑，相比Softmax不需要指数计算，适合稀疏表示
elu1p = lambda x: F.elu(x) + 1

def build_topk_mask(embs: Union[torch.Tensor, np.ndarray], k: int = 768, dim: int = -1):
    """
    构建Top-K掩码：仅保留每个样本中最大的K个维度
    
    这是VDR实现稀疏化的核心工具。通过只保留top-k个最重要的词汇维度，
    VDR实现了：
    1. 降低计算和存储开销（从BERT的30K维降到768维）
    2. 提高可解释性（只保留最相关的词汇）
    3. 减少噪声（过滤掉不重要的维度）
    
    Args:
        embs: 嵌入向量 [batch_size, vocab_size] 或 numpy数组
        k: 保留的top-k个维度数量。默认768（与BERT hidden_size一致）
        dim: 沿着哪个维度选择top-k。默认-1（最后一个维度）
        
    Returns:
        topk_mask: 布尔掩码，形状与embs相同，top-k位置为True，其他为False
    
    例如：
        embs = [[0.1, 0.5, 0.3, 0.9, 0.2]]  # 形状: [1, 5]
        mask = build_topk_mask(embs, k=2)   # 保留最大的2个
        # 结果: [[False, True, False, True, False]]  # 保皙0.5和0.9
    """        
    if isinstance(embs, np.ndarray):
        embs = torch.from_numpy(embs)
    # 获取top-k的值和索引
    values, indices = torch.topk(embs, k, dim=dim)
    # 创建全False的mask
    topk_mask = torch.zeros_like(embs, dtype=torch.bool)
    # 将top-k位置设为True
    topk_mask.scatter_(dim=-1, index=indices, value=True)
    return topk_mask

def topk_sparsify(emb_dense: torch.Tensor, k: int, dim: int = -1):
    """
    对稀密向量进行top-k稀疏化
    
    直接将非 top-k 的维度置为 0，实现稀疏化。
    
    Args:
        emb_dense: 稀密向量 [batch_size, vocab_size]
        k: 保留的维度数
        dim: 稀疏化的维度
    
    Returns:
        emb_sparse: 稀疏向量，仅有k个非零元素
    """
    topk_mask = build_topk_mask(emb_dense, k=k, dim=dim)
    emb_sparse = emb_dense * topk_mask
    return emb_sparse

def build_bow_mask(text_ids, vocab_size=30522, shift_num=0, norm=False):
    """
    构建词袋(Bag-of-Words)掩码：标记输入中出现过的token
    
    BOW表示是VDR的基础组件：
    1. 提供词汇层面的精确匹配（Lexical Matching）
    2. 保证输入的token一定被激活
    3. 与语义表示（Top-K mask）结合，实现半参数化检索
    
    Args:
        text_ids: token ID序列 [batch_size, seq_len]
        vocab_size: 词汇表大小。BERT默认30522
        shift_num: 跳过的特殊token数量。默认0，建议999跳过标点符号
        norm: 是否对BOW向量进行L2归一化
    
    Returns:
        bow_mask: 二值或浮点掩码 [batch_size, vocab_size - shift_num]
                  对于输入中出现的token，对应位置为1，其他为0
    
    例如：
        text_ids = [[101, 2054, 2003]]  # [CLS] what is
        # 输出: [[0, 0, ..., 1, 0, ..., 1, ...]]  # 仅在2054和2003位置为1
    """
    N = text_ids.shape[0]
    V = vocab_size
    # 使用scatter_将text_ids中的token位置设为1
    # 初始化全零矩阵 [N, V]
    bow_mask = torch.zeros([N, V]).to(text_ids.device).scatter_(-1, text_ids, 1).bool().float()
    
    # 如果要跳过前 shift_num 个 token（特殊符号和标点）
    if shift_num > 0:
        bow_mask = bow_mask[:, shift_num:].contiguous()
    
    # 可选：L2归一化
    if norm:
        bow_mask = F.normalize(bow_mask)
    return bow_mask


def init_cts_mask_like(embs):
    """
    初始化对比(Contrastive)掩码的基础结构
    
    为每个样本分配一个“专属”的词汇子集，用于对比学习。
    这是一种稀疏的负采样策略，避免与其他样本的词汇冲突。
    
    Args:
        embs: 嵌入向量 [batch_size, vocab_size]
    
    Returns:
        cts_mask: 对比掩码 [batch_size, vocab_size]
                  每个样本有不同的“专属”词汇子集
    
    原理：使用模运算将词汇空间分配给不同的样本
    例如batch_size=4，vocab_size=8:
        样本0分配：[0, 4, 8, ...]  # vocab_idx % 4 == 0
        样本1分配：[1, 5, 9, ...]  # vocab_idx % 4 == 1
        ...
    """
    batch_size, vocab_size = embs.size()
    # 使用模运算分配词汇
    indices = torch.arange(vocab_size) % batch_size
    # 构建对比掩码：每个样本只关注自己的词汇子集
    cts_mask = (indices.unsqueeze(0) == torch.arange(batch_size).unsqueeze(1))
    return cts_mask.to(embs.device)


def build_cts_mask(bow_embs):
    """
    构建对比学习的掩码：避免与已出现词汇冲突
    
    在VDR的对比学习中，我们需要确保：
    1. 每个样本有一些“专属”的词汇维度用于对比
    2. 这些维度不与输入中已出现的词汇冲突
    
    这样可以：
    - 增强负采样的多样性
    - 避免词汇表冲突（同一个词同时作为正负样本）
    - 提高训练稳定性
    
    Args:
        bow_embs: BOW表示 [batch_size, vocab_size]
                  标记每个样本中出现的词汇
    
    Returns:
        cts_mask: 对比掩码 [batch_size, vocab_size]
                  True表示该维度可用于此样本的对比学习
    
    原理：
        cts_mask = (初始对比掩码) AND NOT (所有样本的BOW并集)
        即：只激活专属的且未出现过的词汇维度
    """
    # 计算所有样本中出现过的词汇（沿着batch维度求并集）
    bow_batch = bow_embs.sum(0).bool()  # [vocab_size]
    
    # 初始化对比掩码
    cts_mask_init = init_cts_mask_like(bow_embs)  # [batch_size, vocab_size]
    
    # 只保留未出现过的词汇
    # cts_mask_init: 专属词汇  AND  ~bow_batch: 未出现的词汇
    cts_mask = (cts_mask_init & ~bow_batch.unsqueeze(0))
    
    return cts_mask.bool().to(bow_embs.device)




