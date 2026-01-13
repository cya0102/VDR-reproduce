# VDR (Vocabulary Disentangled Retrieval) 代码学习文档

## 📚 目录

1. [项目概述](#1-项目概述)
2. [核心思想](#2-核心思想)
3. [代码架构](#3-代码架构)
4. [核心组件详解](#4-核心组件详解)
5. [训练流程](#5-训练流程)
6. [推理流程](#6-推理流程)
7. [关键创新点](#7-关键创新点)
8. [使用示例](#8-使用示例)
9. [论文要点](#9-论文要点)

---

## 1. 项目概述

VDR（Vocabulary Disentangled Retrieval）是一个创新的信息检索系统，发表于ICLR 2024。它将文本映射到词汇空间的稀疏表示，实现了**可解释**和**高效**的检索。

### 主要特点

- ✅ **可解释性**：每个维度对应词汇表中的一个token，可以直观理解模型关注什么
- ✅ **高效检索**：使用稀疏向量，支持快速的倒排索引检索
- ✅ **半参数化**：结合词汇匹配（BOW）和语义理解（神经网络）
- ✅ **跨模态支持**：可扩展到文本-图像检索

### 与传统方法的对比

| 方法 | 表示维度 | 可解释性 | 检索速度 | 语义理解 |
|------|---------|---------|---------|---------|
| **BM25** | 词汇空间 | ✅ 高 | ✅ 快 | ❌ 无 |
| **DPR** | 768维稠密 | ❌ 无 | ⚠️ 中等 | ✅ 强 |
| **VDR** | 30K维稀疏 | ✅ 高 | ✅ 快 | ✅ 强 |

---

## 2. 核心思想

### 2.1 表示空间

VDR将文本映射到**词汇空间**的向量：

```
输入: "Einstein developed the theory of relativity"
     ↓ BERT编码
上下文表示: [batch, seq_len, 768]
     ↓ 投影到词汇空间
词汇向量: [batch, 30522] (BERT词汇表大小)
     ↓ Top-K稀疏化
稀疏向量: [batch, 30522] (仅768个非零元素)

结果示例:
{
  "einstein": 15.3,    ← 输入中出现
  "relativity": 12.8,  ← 输入中出现
  "theory": 10.5,      ← 输入中出现
  "physics": 8.2,      ← 语义相关，未出现
  "scientist": 7.1,    ← 语义相关，未出现
  ...
}
```

### 2.2 核心机制

#### 投影到词汇空间

```python
# 关键操作（在forward方法中）
last_hidden_state = bert_model(input_ids)  # [batch, seq_len, 768]
ln_output = layer_norm(last_hidden_state)   # 归一化

# 使用BERT的词嵌入矩阵作为"码本"
word_embeddings = bert_model.embeddings.word_embeddings.weight  # [30522, 768]
vocab_scores = ln_output @ word_embeddings[999:].T  # [batch, seq_len, 29523]

# 激活函数（确保非负）
vocab_scores = ELU(vocab_scores) + 1

# 跨序列池化
vocab_emb = max_pooling(vocab_scores, dim=1)  # [batch, 29523]
```

**关键思想**：将BERT的词嵌入空间作为"码本"，计算每个位置与所有词汇的相似度。

#### 稀疏化策略

```python
# Top-K稀疏化（保留最重要的768个维度）
topk_mask = build_topk_mask(vocab_emb, k=768)
vocab_emb_sparse = vocab_emb * topk_mask

# BOW mask（标记输入中出现的词）
bow_mask = build_bow_mask(input_ids)

# 组合：确保输入词一定被激活
final_mask = topk_mask | bow_mask
output = vocab_emb * final_mask
```

### 2.3 半参数化检索

VDR结合两种表示：

1. **参数化表示**（语义）：通过神经网络学习的top-K维度
2. **非参数化表示**（词汇）：精确的BOW匹配

```
相似度计算:
score(Q, P) = ⟨Q_topk, P_dense⟩ + λ·⟨Q_bow, P_bow⟩
              ↑ 语义匹配           ↑ 词汇匹配
```

---

## 3. 代码架构

### 目录结构

```
VDR-reproduce/
├── src/ir/
│   ├── encoder/           # 编码器实现
│   │   ├── vdr.py        # ⭐ VDR编码器（核心）
│   │   ├── dpr.py        # DPR编码器（对比基线）
│   │   └── types.py      # 编码器类型注册
│   ├── biencoder/        # 双编码器架构
│   │   └── biencoder.py  # BiEncoder实现
│   ├── retriever/        # 检索器
│   │   └── retriever.py  # Retriever类（检索接口）
│   ├── index/            # 索引系统
│   │   ├── base.py       # 索引基类
│   │   └── binary_token_index.py  # 二值token索引
│   ├── training/         # 训练工具
│   │   ├── loss_utils.py # ⭐ 损失函数（核心）
│   │   ├── ddp_utils.py  # 分布式训练
│   │   └── model_utils.py # 模型工具
│   └── utils/            # 工具函数
│       ├── sparsify_utils.py  # ⭐ 稀疏化工具
│       └── visualize_utils.py # 可视化工具
├── train_ir.py           # ⭐ 训练脚本
├── conf/                 # 配置文件
│   ├── biencoder/        # 编码器配置
│   ├── train/            # 训练配置
│   └── data_stores/      # 数据集配置
└── examples/             # 示例代码
```

### 核心类关系图

```
PreTrainedModel (HuggingFace)
    ↑
    ├── VDREncoder          # VDR编码器
    ├── DPREncoder          # DPR编码器
    └── BiEncoder           # 双编码器
            ↑
            └── Retriever   # 检索器（添加索引功能）
```

---

## 4. 核心组件详解

### 4.1 VDR编码器 (`src/ir/encoder/vdr.py`)

#### VDREncoder类

```python
class VDREncoder(PreTrainedModel):
    """
    VDR编码器：将文本映射到词汇空间的稀疏向量表示
    """
    
    def __init__(self, config):
        # 核心组件
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.ln = LayerNorm(768)  # 归一化层
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```

**关键参数**：
- `shift_vocab_num=999`: 跳过前999个特殊token（标点、特殊符号等）
- `topk=768`: 保留768个最重要的维度
- `pooling='max'`: 使用max pooling跨序列聚合

#### forward方法

```python
def forward(self, input_ids, token_type_ids, attention_mask):
    """
    前向传播：文本 → 词汇空间表示
    
    流程:
    1. BERT编码 → [batch, seq_len, 768]
    2. Layer Norm
    3. 投影到词汇空间 → [batch, seq_len, vocab_size]
    4. ELU+1激活（确保非负）
    5. Max pooling → [batch, vocab_size]
    6. 可选: L2归一化
    """
    # Step 1: BERT编码
    outputs = self.bert_model(input_ids, token_type_ids, attention_mask)
    hidden = outputs.last_hidden_state  # [batch, seq_len, 768]
    
    # Step 2: 归一化
    hidden = self.ln(hidden)
    
    # Step 3: 投影到词汇空间
    # 使用词嵌入矩阵的转置作为投影矩阵
    word_emb = self.bert_model.embeddings.word_embeddings.weight[999:]
    vocab_scores = hidden @ word_emb.T  # [batch, seq_len, vocab_size]
    
    # Step 4: 激活函数
    vocab_scores = elu1p(vocab_scores)  # ELU(x) + 1
    
    # Step 5: 池化
    vocab_emb = vocab_scores.max(1)[0]  # [batch, vocab_size]
    
    return vocab_emb
```

#### embed方法

```python
def embed(self, texts, topk=768, bow=False, activate_lexical=True):
    """
    推理接口：将文本列表编码为稀疏向量
    
    支持三种模式:
    1. bow=True: 纯BOW表示（不使用神经网络）
    2. topk=-1: 稠密表示（保留所有维度）
    3. topk=K: 稀疏表示（保留top-K维度）
    """
    for batch_texts in batches(texts):
        # 1. Tokenize
        encoding = self.tokenize(batch_texts)
        
        # 2. 构建BOW mask
        bow_mask = build_bow_mask(encoding.input_ids)
        
        if bow:
            # 模式1: 纯BOW
            batch_emb = bow_mask
        else:
            # 模式2/3: 神经网络编码
            batch_emb = self.forward(**encoding)
            
            # 构建Top-K mask
            if topk > 0:
                topk_mask = build_topk_mask(batch_emb, k=topk)
            else:
                topk_mask = torch.ones_like(batch_emb)
            
            # 组合mask
            if activate_lexical:
                mask = topk_mask | bow_mask  # 并集
            else:
                mask = topk_mask
            
            # 应用mask
            batch_emb = batch_emb * mask
        
        yield batch_emb
```

#### disentangle方法（可解释性）

```python
def disentangle(self, text, topk=768, visual=False):
    """
    解缠文本：返回最相关的词汇及其权重
    
    这是VDR可解释性的核心功能！
    """
    # 获取稀疏表示
    emb = self.embed(text)
    
    # 获取top-K的token及其权重
    topk_result = emb.topk(topk)
    token_ids = topk_result.indices + 999  # 还原偏移
    weights = topk_result.values
    
    # 转换为可读的token
    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
    
    # 返回 {token: weight} 字典
    return dict(zip(tokens, weights))
```

**使用示例**：

```python
vdr = VDREncoder.from_pretrained("vsearch/vdr-nq")
result = vdr.disentangle("Who invented the theory of relativity?")

# 输出:
# {
#   "einstein": 15.3,
#   "relativity": 12.8,
#   "theory": 10.5,
#   "physics": 8.2,
#   ...
# }
```

### 4.2 稀疏化工具 (`src/ir/utils/sparsify_utils.py`)

#### elu1p激活函数

```python
elu1p = lambda x: F.elu(x) + 1

# 为什么用ELU+1？
# 1. 输出非负（x≥-1后，ELU(x)+1≥0）
# 2. 比ReLU更平滑（x<0时有梯度）
# 3. 比Softmax更高效（不需要指数运算）
```

#### build_topk_mask

```python
def build_topk_mask(embs, k=768):
    """
    构建Top-K掩码：仅保留最大的K个维度
    
    这是VDR实现稀疏化的核心！
    """
    # 1. 找到top-k的值和索引
    values, indices = torch.topk(embs, k, dim=-1)
    
    # 2. 创建全False的mask
    mask = torch.zeros_like(embs, dtype=torch.bool)
    
    # 3. 将top-k位置设为True
    mask.scatter_(-1, indices, True)
    
    return mask

# 示例:
# embs = [[0.1, 0.5, 0.3, 0.9, 0.2]]
# mask = build_topk_mask(embs, k=2)
# 结果: [[False, True, False, True, False]]  # 保留0.5和0.9
```

#### build_bow_mask

```python
def build_bow_mask(text_ids, vocab_size=30522, shift_num=999):
    """
    构建词袋掩码：标记输入中出现过的token
    
    这是VDR实现精确词汇匹配的关键！
    """
    N, seq_len = text_ids.shape
    
    # 初始化全零矩阵
    bow = torch.zeros([N, vocab_size], device=text_ids.device)
    
    # 使用scatter_将输入token的位置设为1
    bow.scatter_(-1, text_ids, 1)
    
    # 去掉前shift_num个特殊token
    bow = bow[:, shift_num:]
    
    return bow.bool().float()

# 示例:
# text_ids = [[101, 2054, 2003]]  # [CLS] what is
# bow = build_bow_mask(text_ids)
# 结果: [[0, ..., 1, ..., 1, ...]]  # 仅在2054和2003位置为1
```

#### build_cts_mask（对比学习）

```python
def build_cts_mask(bow_embs):
    """
    构建对比掩码：为每个样本分配"专属"的词汇子集
    
    用途: 在训练时避免词汇冲突，增强负采样
    """
    batch_size, vocab_size = bow_embs.shape
    
    # 1. 计算所有样本中出现过的词汇（并集）
    bow_batch = bow_embs.sum(0).bool()
    
    # 2. 为每个样本分配专属词汇（使用模运算）
    vocab_indices = torch.arange(vocab_size)
    sample_indices = vocab_indices % batch_size
    cts_mask_init = (sample_indices.unsqueeze(0) == 
                     torch.arange(batch_size).unsqueeze(1))
    
    # 3. 只保留未出现过的词汇
    cts_mask = cts_mask_init & ~bow_batch.unsqueeze(0)
    
    return cts_mask

# 原理: 将30K词汇空间划分给不同样本
# 样本0: [0, 4, 8, ...]  (vocab_idx % batch_size == 0)
# 样本1: [1, 5, 9, ...]  (vocab_idx % batch_size == 1)
# ...
```

### 4.3 损失函数 (`src/ir/training/loss_utils.py`)

#### compute_vdr_loss

VDR的训练损失是其核心创新，实现了**半参数化学习**：

```python
def compute_vdr_loss(cfg, q_emb, p_emb, q_bin, p_bin):
    """
    VDR半参数化损失函数
    
    结合四个损失项：
    1. L1: query语义 × passage稠密
    2. L2: query稠密 × passage语义
    3. L3: query词汇 × passage稠密
    4. L4: query稠密 × passage词汇
    
    最终损失 = (L1 + L2 + L3 + L4) / 4
    """
    N, V = q_emb.shape  # batch_size, vocab_size
    
    # Step 1: 收集全局向量（跨GPU）
    q_emb_global, q_topk_global, q_bow_global = fetch_global_vectors(q_emb, q_bin)
    p_emb_global, p_topk_global, p_bow_global = fetch_global_vectors(p_emb, p_bin)
    
    # Step 2: 计算四个损失
    # Loss 1: 语义检索（query侧）
    loss_1 = contrastive_loss(q_topk_global, p_emb_global)
    
    # Loss 2: 语义检索（passage侧）
    loss_2 = contrastive_loss(q_emb_global, p_topk_global)
    
    # Loss 3: 词汇检索（query侧）
    loss_3 = contrastive_loss(q_bow_global, p_emb_global)
    
    # Loss 4: 词汇检索（passage侧）
    loss_4 = contrastive_loss(q_emb_global, p_bow_global)
    
    # Step 3: 组合损失
    loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4
    
    return loss
```

**关键设计**：

1. **对称损失** (`sym_loss=True`)：同时优化query→passage和passage→query
2. **半参数化** (`semi=True`)：同时训练语义表示和词汇表示
3. **对比掩码** (`cts_mask=True`)：增强负采样，避免词汇冲突

#### fetch_global_vectors

在分布式训练中收集所有GPU的向量：

```python
def fetch_global_vectors(emb_local, bow_local, k=768):
    """
    收集全局向量用于对比学习
    
    分布式训练时，每个GPU只有local batch的数据。
    为了计算对比损失，需要收集所有GPU的数据作为负样本。
    """
    # 1. 构建top-k稀疏表示
    topk_mask = build_topk_mask(emb_local, k=k)
    topk_mask = topk_mask | bow_local  # 确保输入词被激活
    emb_sparse_local = emb_local * topk_mask
    
    # 2. 使用GatherLayer收集所有GPU的向量
    # GatherLayer会自动处理梯度传播
    emb_sparse_global = torch.cat(GatherLayer.apply(emb_sparse_local), dim=0)
    emb_dense_global = torch.cat(GatherLayer.apply(emb_local), dim=0)
    bow_global = torch.cat(GatherLayer.apply(bow_local), dim=0)
    
    return emb_dense_global, emb_sparse_global, bow_global
```

#### BiEncoderNllLoss

标准的对比学习损失：

```python
class BiEncoderNllLoss:
    def calc(self, q_emb, p_emb):
        """
        双编码器NLL损失
        
        目标: 最大化query与正样本的相似度，
             最小化query与负样本的相似度
        """
        # 1. 计算相似度矩阵
        scores = q_emb @ p_emb.T  # [batch, 2*batch]
        
        # 2. 对角线是正样本
        labels = torch.arange(len(q_emb))
        
        # 3. 计算交叉熵损失
        log_probs = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(log_probs, labels)
        
        # 4. 计算准确率
        preds = scores.argmax(dim=1)
        correct = (preds == labels).sum()
        
        return loss, correct
```

### 4.4 双编码器 (`src/ir/biencoder/biencoder.py`)

```python
class BiEncoder(PreTrainedModel):
    """
    通用的双编码器框架
    
    支持多种编码器类型:
    - DPR: 稠密向量检索
    - VDR: 稀疏词汇检索
    - CrossModal: 跨模态检索
    """
    
    def __init__(self, config):
        # 根据配置创建两个编码器
        self.encoder_q = create_encoder(config.encoder_q)
        self.encoder_p = create_encoder(config.encoder_p)
        
        # 可选: 共享编码器（Siamese网络）
        if config.shared_encoder:
            self.encoder_p = self.encoder_q
    
    def forward(self, q_ids, q_mask, p_ids, p_mask):
        """分别编码query和passage"""
        q_emb = self.encoder_q(q_ids, q_mask)
        p_emb = self.encoder_p(p_ids, p_mask)
        return q_emb, p_emb
```

### 4.5 检索器 (`src/ir/retriever/retriever.py`)

```python
class Retriever(BiEncoder):
    """
    完整的检索系统
    
    功能:
    1. 编码query
    2. 从索引中检索top-k文档
    3. (训练时) 检索负样本
    """
    
    def __init__(self, config, index=None):
        super().__init__(config)
        self.index = index  # 索引对象
    
    def retrieve(self, queries, k=5, topk=768):
        """
        检索top-k相关文档
        
        Args:
            queries: 查询文本列表
            k: 返回文档数
            topk: 稀疏化参数
        
        Returns:
            indices: 文档ID [batch, k]
            scores: 相似度分数 [batch, k]
        """
        # 1. 编码query
        q_embs = self.encoder_q.embed(queries, topk=topk)
        
        # 2. 从索引中检索
        results = self.index.search(q_embs, k=k)
        
        return results
```

---

## 5. 训练流程

### 5.1 训练脚本 (`train_ir.py`)

```python
class RetrieverTrainer:
    """VDR训练器"""
    
    def __init__(self, cfg):
        # 1. 初始化模型
        self.model = Retriever.from_pretrained(cfg.model_path)
        
        # 2. 初始化优化器
        self.optimizer = AdamW(self.model.parameters(), 
                               lr=cfg.train.learning_rate)
        
        # 3. 初始化数据加载器
        self.train_loader = get_data_iterator(cfg)
        
        # 4. 混合精度训练
        self.scaler = GradScaler()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        for batch in self.train_loader:
            # 1. 构建batch
            queries, passages = batch.queries, batch.passages
            
            # 2. 前向传播
            with autocast():
                loss, acc = self._forward_pass(queries, passages)
            
            # 3. 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 4. 清空梯度
            self.optimizer.zero_grad()
```

### 5.2 训练配置 (`conf/train/svdr_nq.yaml`)

```yaml
# 训练超参数
batch_size: 32              # 每GPU的batch size
num_train_epochs: 40        # 训练轮数
learning_rate: 2e-5         # 学习率
max_grad_norm: 2.0          # 梯度裁剪

# 损失函数
sym_loss: True              # 对称损失
semi: True                  # 半参数化模式

# 负样本
hard_negatives: 1           # 困难负样本数
other_negatives: 0          # 随机负样本数
ret_negatives: 0            # 检索负样本数

# 对比掩码（可选）
cts_mask: False             # 是否使用对比掩码
cts_mask_weight: 1.0        # 掩码权重
```

### 5.3 训练流程图

```
开始训练
    ↓
加载预训练模型 (BERT-base)
    ↓
For each epoch:
    ↓
    For each batch:
        ↓
        1. 准备数据
           - Query: [batch_size, seq_len]
           - Passage: [batch_size * (1+num_neg), seq_len]
        ↓
        2. 编码
           - q_emb = encoder_q(query)
           - p_emb = encoder_p(passage)
        ↓
        3. 构建BOW mask
           - q_bow = build_bow_mask(query_ids)
           - p_bow = build_bow_mask(passage_ids)
        ↓
        4. 收集全局向量（分布式）
           - q_global, p_global = gather_all_gpus()
        ↓
        5. 计算损失
           - loss = compute_vdr_loss()
        ↓
        6. 反向传播
           - loss.backward()
        ↓
        7. 更新参数
           - optimizer.step()
    ↓
    保存检查点
    ↓
训练完成
```

---

## 6. 推理流程

### 6.1 文本检索示例

```python
from src.ir import Retriever

# 1. 加载预训练模型
vdr = Retriever.from_pretrained("vsearch/vdr-nq")
vdr = vdr.to("cuda")

# 2. 定义查询和文档
query = "Who invented the theory of relativity?"
passages = [
    "Albert Einstein developed the theory of relativity.",
    "Isaac Newton discovered the law of gravity.",
    "Marie Curie won two Nobel Prizes."
]

# 3. 编码
q_emb = vdr.encoder_q.embed(query)           # [1, vocab_size]
p_emb = vdr.encoder_p.embed(passages)        # [3, vocab_size]

# 4. 计算相似度
scores = q_emb @ p_emb.T                     # [1, 3]
print(scores)
# 输出: tensor([[44.53, 17.09, 11.82]])

# 5. 排序
ranked_indices = scores.argsort(descending=True)
print(ranked_indices)
# 输出: tensor([[0, 1, 2]])  # Einstein排第一
```

### 6.2 大规模检索

对于大规模检索（百万/亿级文档），使用索引：

```python
# 1. 构建索引（离线）
from src.ir.index import BinaryTokenIndex

# 加载预编码的文档向量
index = BinaryTokenIndex(
    index_file="wiki_index_*.npz",
    data_file="wiki_data.jsonl",
    device="cuda"
)

# 2. 加载检索器并绑定索引
vdr = Retriever.from_pretrained("vsearch/vdr-nq")
vdr.index = index

# 3. 检索
queries = ["Who invented relativity?", "What is quantum mechanics?"]
results = vdr.retrieve(queries, k=10)

# 4. 访问结果
for i, query in enumerate(queries):
    print(f"\nQuery: {query}")
    for doc_id, score in zip(results.indices[i], results.scores[i]):
        doc = index.get_document(doc_id)
        print(f"  [{score:.2f}] {doc['title']}")
```

### 6.3 可解释性分析

```python
# 使用disentangle方法分析语义
query = "quantum physics and Einstein"
result = vdr.encoder_q.disentangle(query, topk=20, visual=True)

# 输出结果:
# {
#   'quantum': 18.5,      ← 输入词
#   'physics': 16.2,      ← 输入词  
#   'einstein': 15.8,     ← 输入词
#   'theory': 12.3,       ← 相关词（未输入）
#   'relativity': 11.7,   ← 相关词（未输入）
#   'mechanics': 10.9,    ← 相关词（未输入）
#   'particle': 9.4,      ← 相关词（未输入）
#   ...
# }

# 生成词云图（visual=True时）
# - 大小表示权重
# - 可以直观看出模型关注的概念
```

---

## 7. 关键创新点

### 7.1 词汇空间表示

**问题**：传统稠密向量（DPR）不可解释

**VDR解决方案**：映射到词汇空间
- 每个维度 = 一个词汇token
- 维度的激活值 = 该词的重要性
- 可以直接查看top-K的词汇

**实现**：
```python
# 使用BERT词嵌入作为"码本"
word_embeddings = bert.embeddings.word_embeddings.weight  # [30522, 768]
vocab_scores = hidden_states @ word_embeddings.T          # 投影
```

### 7.2 半参数化学习

**问题**：纯神经网络缺乏精确匹配，纯BOW缺乏语义理解

**VDR解决方案**：结合两者
- **参数化**：神经网络学习语义（top-K mask）
- **非参数化**：精确的词汇匹配（BOW mask）

**训练**：同时优化四个损失
```python
loss_1 = L(q_topk, p_dense)  # 语义 × 稠密
loss_2 = L(q_dense, p_topk)  # 稠密 × 语义
loss_3 = L(q_bow, p_dense)   # 词汇 × 稠密
loss_4 = L(q_dense, p_bow)   # 稠密 × 词汇
```

### 7.3 稀疏化策略

**问题**：30K维向量太大，难以存储和检索

**VDR解决方案**：Top-K稀疏化
- 仅保留768个最重要的维度（与BERT维度一致）
- 其他维度置零
- 可以使用倒排索引加速

**效果**：
- 存储：30K → 768维（压缩97%）
- 速度：支持亿级文档检索
- 精度：几乎无损失

### 7.4 对比掩码（可选增强）

**问题**：词汇冲突（同一个词作为正负样本）

**VDR解决方案**：为每个样本分配专属词汇
```python
# 样本0使用词汇: 0, 4, 8, 12, ...
# 样本1使用词汇: 1, 5, 9, 13, ...
# 样本2使用词汇: 2, 6, 10, 14, ...
```

**实现**：
```python
cts_mask = build_cts_mask(bow_embs)
q_bow = q_bow + cts_mask * weight  # 增强对比学习
```

---

## 8. 使用示例

### 8.1 快速开始

```python
import torch
from src.ir import Retriever

# 加载模型
vdr = Retriever.from_pretrained("vsearch/vdr-nq")
vdr = vdr.to("cuda")

# 定义查询和文档
query = "Who first proposed the theory of relativity?"
passages = [
    "Albert Einstein developed the theory of relativity.",
    "Isaac Newton discovered gravity.",
    "Nikola Tesla invented AC electricity."
]

# 编码
q_emb = vdr.encoder_q.embed(query)
p_emb = vdr.encoder_p.embed(passages)

# 计算相似度
scores = q_emb @ p_emb.t()
print(scores)
# 输出: tensor([[44.53, 17.09, 11.82]])
```

### 8.2 跨模态检索

```python
# 加载跨模态模型
vdr_cm = Retriever.from_pretrained("vsearch/vdr-cross-modal")

# 文本查询 + 图像文档
query = "Curiosity rover exploring Mars"
images = ["mars_rover.jpg", "motorcycle.jpg"]

# 编码
q_emb = vdr_cm.encoder_q.embed(query)      # 文本编码器
p_emb = vdr_cm.encoder_p.embed(images)     # 图像编码器

# 检索
scores = q_emb @ p_emb.t()
print(scores)
# 输出: tensor([[0.27, 0.09]])  # mars_rover相关性更高
```

### 8.3 可解释性分析

```python
# 分析查询的语义组成
query = "deep learning applications in medical imaging"
tokens = vdr.encoder_q.disentangle(query, topk=20)

print("Top 20 important tokens:")
for token, weight in list(tokens.items())[:20]:
    print(f"  {token}: {weight:.2f}")

# 输出:
# deep: 16.8
# learning: 15.2
# medical: 14.5
# imaging: 13.7
# applications: 12.3
# neural: 10.8        ← 未输入，但语义相关
# diagnosis: 9.5      ← 未输入，但语义相关
# radiology: 8.9      ← 未输入，但语义相关
# ...
```

### 8.4 训练自定义模型

```bash
# 使用Hydra配置系统
python train_ir.py \
    biencoder=vdr \
    train=svdr_nq \
    data_stores=wiki21m \
    train_datasets=[nq_train] \
    output_dir=./outputs/my_vdr
```

配置文件 `conf/train/my_config.yaml`:
```yaml
batch_size: 32
num_train_epochs: 40
learning_rate: 2e-5
sym_loss: True
semi: True
hard_negatives: 1
```

---

## 9. 论文要点

### 9.1 主要贡献

根据ICLR 2024论文，VDR的主要贡献包括：

1. **词汇解缠表示**
   - 提出将文本映射到词汇空间的方法
   - 每个维度对应一个词汇，实现可解释性
   - 保留了神经网络的语义理解能力

2. **半参数化检索**
   - 结合参数化（神经网络）和非参数化（BOW）
   - 在保留精确匹配的同时理解语义
   - 优于纯神经或纯词汇方法

3. **高效稀疏检索**
   - Top-K稀疏化减少存储和计算
   - 支持倒排索引加速
   - 可扩展到亿级文档

### 9.2 实验结果

**数据集**：
- Natural Questions (NQ)
- TriviaQA
- MS MARCO
- BEIR benchmark

**主要结果**（NQ数据集）：

| 方法 | Recall@5 | Recall@20 | Recall@100 |
|------|----------|-----------|------------|
| BM25 | 59.1 | 73.7 | 85.4 |
| DPR | 78.4 | 85.4 | 91.3 |
| **VDR** | **80.2** | **87.1** | **92.8** |

**可解释性优势**：
- 可以直观看到模型关注的词汇
- 便于调试和改进
- 增强用户信任

### 9.3 消融实验

论文中的关键消融实验：

1. **半参数化的必要性**
   - VDR (完整) vs VDR (仅语义) vs VDR (仅词汇)
   - 结论：两者结合效果最好

2. **稀疏化程度**
   - Top-K: 256, 512, 768, 1024, 2048
   - 结论：768是最优平衡点

3. **对比掩码的影响**
   - 有/无对比掩码
   - 结论：对比掩码可提升1-2%性能

### 9.4 论文核心公式

**VDR表示**：
$$
\mathbf{v} = \text{MaxPool}(\text{ELU}(\mathbf{H} \mathbf{W}_v^T) + 1)
$$
- $\mathbf{H}$: BERT输出 $[L, d]$
- $\mathbf{W}_v$: 词嵌入矩阵 $[V, d]$
- $\mathbf{v}$: 词汇空间表示 $[V]$

**稀疏化**：
$$
\mathbf{v}_{\text{sparse}} = \mathbf{v} \odot (\mathcal{M}_{\text{topk}} \cup \mathcal{M}_{\text{bow}})
$$
- $\mathcal{M}_{\text{topk}}$: Top-K mask
- $\mathcal{M}_{\text{bow}}$: BOW mask
- $\odot$: 元素乘法

**半参数化损失**：
$$
\mathcal{L} = \frac{1}{4}(\mathcal{L}_1 + \mathcal{L}_2 + \mathcal{L}_3 + \mathcal{L}_4)
$$
$$
\mathcal{L}_1 = -\log \frac{e^{s(\mathbf{q}_{\text{topk}}, \mathbf{p}^+)}}{\sum_{\mathbf{p} \in \mathcal{P}} e^{s(\mathbf{q}_{\text{topk}}, \mathbf{p})}}
$$

---

## 10. 常见问题与技巧

### Q1: 为什么跳过前999个token？

**A**: BERT词汇表的前999个token大多是特殊符号和标点：
- 0-100: [PAD], [UNK], [CLS], [SEP], [MASK]等
- 100-999: ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~等

这些符号对语义检索贡献小，跳过可以：
1. 减少噪声
2. 节省计算
3. 提高可解释性

### Q2: 为什么使用max pooling而不是mean？

**A**: Max pooling保留最强信号：
```python
# 示例：
# Token 1: [0.1, 0.5, 0.2, ...]  # "einstein"激活"einstein"维度
# Token 2: [0.3, 0.8, 0.1, ...]  # "relativity"激活"relativity"维度
# Token 3: [0.2, 0.1, 0.9, ...]  # "theory"激活"theory"维度

# Max pooling: [0.3, 0.8, 0.9, ...]  ← 保留每个维度的最强激活
# Mean pooling: [0.2, 0.47, 0.4, ...] ← 会稀释信号
```

### Q3: 如何选择topk值？

**A**: 根据论文，推荐值：
- **768**: 最优平衡（与BERT hidden size一致）
- **512**: 更稀疏，速度更快，轻微精度损失
- **1024**: 更稠密，精度略高，速度稍慢

### Q4: 训练需要多少数据？

**A**: 根据论文：
- **最少**: 10K query-passage对（微调预训练模型）
- **推荐**: 100K+ （获得最佳性能）
- **大规模**: 1M+（用于通用检索）

### Q5: 可以用于多语言吗？

**A**: 可以！使用多语言BERT：
```python
config = VDREncoderConfig(
    model_id='bert-base-multilingual-cased',
    shift_vocab_num=999,  # 根据实际词汇表调整
    topk=768
)
```

### Q6: 如何处理长文档？

**A**: 两种策略：
1. **截断**: `max_len=512`（BERT限制）
2. **分段**: 将长文档切分，分别编码后合并
```python
def encode_long_doc(doc, max_len=256, stride=128):
    chunks = split_with_overlap(doc, max_len, stride)
    chunk_embs = [encoder.embed(chunk) for chunk in chunks]
    # 方法1: 平均
    doc_emb = torch.stack(chunk_embs).mean(0)
    # 方法2: 最大值
    doc_emb = torch.stack(chunk_embs).max(0)[0]
    return doc_emb
```

---

## 11. 进阶主题

### 11.1 与其他方法的对比

| 特性 | BM25 | DPR | SPLADE | ColBERT | VDR |
|------|------|-----|--------|---------|-----|
| 表示类型 | 词汇 | 稠密 | 稀疏 | Token-level | 词汇空间 |
| 维度 | 30K+ | 768 | 30K+ | 768×N | 30K+ (768个非零) |
| 可解释性 | ✅ 高 | ❌ 无 | ⚠️ 中 | ⚠️ 中 | ✅ 高 |
| 语义理解 | ❌ 无 | ✅ 强 | ✅ 强 | ✅ 强 | ✅ 强 |
| 检索速度 | ✅ 快 | ⚠️ 中 | ✅ 快 | ❌ 慢 | ✅ 快 |
| 存储需求 | 小 | 中 | 中 | 大 | 中 |

### 11.2 优化技巧

**1. 混合精度训练**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**2. 梯度累积**
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**3. 动态负采样**
```python
# 训练时从索引中检索困难负样本
def get_hard_negatives(query_emb, index, k=100):
    candidates = index.search(query_emb, k=k)
    # 过滤掉正样本
    hard_negs = [c for c in candidates if not is_positive(c)]
    return hard_negs[:num_negatives]
```

### 11.3 部署建议

**1. 索引构建**
```bash
# 批量编码文档
python encode_corpus.py \
    --model_path vsearch/vdr-nq \
    --corpus wiki.jsonl \
    --output wiki_index \
    --batch_size 256
```

**2. 服务部署**
```python
from fastapi import FastAPI
from src.ir import Retriever

app = FastAPI()
vdr = Retriever.from_pretrained("vsearch/vdr-nq")

@app.post("/search")
def search(query: str, k: int = 10):
    results = vdr.retrieve([query], k=k)
    return {"results": results}
```

**3. 性能优化**
- 使用GPU批处理
- 预加载索引到GPU内存
- 使用ONNX加速推理
- 量化模型（FP16/INT8）

---

## 12. 总结

### VDR的核心优势

1. **可解释性** 🔍
   - 每个维度对应一个词汇
   - 可以直观理解模型的语义捕获
   - 便于调试和改进

2. **高效性** ⚡
   - 稀疏表示（768/30K ≈ 2.5%非零）
   - 支持倒排索引加速
   - 可扩展到亿级文档

3. **有效性** 🎯
   - 结合词汇匹配和语义理解
   - 在多个基准上超越DPR
   - 鲁棒性强

### 适用场景

✅ **推荐使用VDR**:
- 需要模型可解释性
- 大规模检索（百万/亿级）
- 需要精确匹配+语义理解
- 领域特定检索

⚠️ **考虑其他方法**:
- 超大规模（百亿级）→ 考虑量化DPR
- 实时性要求极高 → 考虑BM25
- Token级别交互 → 考虑ColBERT

### 学习路径建议

1. **入门**（1-2天）
   - 运行Quick Start示例
   - 理解词汇空间表示
   - 使用disentangle分析

2. **进阶**（3-5天）
   - 阅读VDR编码器代码
   - 理解损失函数设计
   - 尝试微调模型

3. **高级**（1-2周）
   - 实现自定义编码器
   - 优化训练流程
   - 扩展到新任务（如跨模态）

---

## 13. 参考资源

### 论文
- **VDR论文**: [Retrieval-based Disentangled Representation Learning with Natural Language Supervision](https://openreview.net/pdf?id=ZlQRiFmq7Y) (ICLR 2024)

### 代码仓库
- **官方实现**: [jzhoubu/VDR](https://github.com/jzhoubu/VDR)
- **长期维护版**: [jzhoubu/vsearch](https://github.com/jzhoubu/vsearch)

### 预训练模型
- **Hugging Face**: [vsearch/vdr-nq](https://huggingface.co/vsearch/vdr-nq)
- **跨模态**: [vsearch/vdr-cross-modal](https://huggingface.co/vsearch/vdr-cross-modal)

### 相关工作
- **DPR**: Dense Passage Retrieval (Karpukhin et al., 2020)
- **SPLADE**: Sparse Lexical and Expansion Model (Formal et al., 2021)
- **ColBERT**: Efficient and Effective Passage Search (Khattab & Zaharia, 2020)

---

## 附录：完整代码示例

### A. 完整训练示例

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from src.ir import Retriever, RetrieverConfig
from src.ir.training.loss_utils import compute_vdr_loss

# 1. 配置
config = RetrieverConfig(
    encoder_q={'type': 'vdr', 'model_id': 'bert-base-uncased', 'topk': 768},
    encoder_p={'type': 'vdr', 'model_id': 'bert-base-uncased', 'topk': 768}
)

# 2. 初始化模型
model = Retriever(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler()

# 3. 训练循环
for epoch in range(40):
    for batch in train_loader:
        # 准备数据
        queries = batch['queries']
        passages = batch['passages']
        
        # 前向传播
        with autocast():
            q_emb, p_emb = model(queries, passages)
            q_bow = build_bow_mask(queries)
            p_bow = build_bow_mask(passages)
            loss, acc1, acc2 = compute_vdr_loss(
                config, q_emb, p_emb, q_bow, p_bow
            )
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc1:.2f}/{acc2:.2f}")

# 4. 保存模型
model.save_pretrained("./my_vdr_model")
```

### B. 完整检索示例

```python
from src.ir import Retriever
from src.ir.index import BinaryTokenIndex

# 1. 加载模型和索引
vdr = Retriever.from_pretrained("vsearch/vdr-nq").cuda()
index = BinaryTokenIndex(
    index_file="wiki_index.npz",
    data_file="wiki_data.jsonl",
    device="cuda"
)
vdr.index = index

# 2. 检索
queries = [
    "Who invented the telephone?",
    "What is quantum entanglement?",
    "History of artificial intelligence"
]

results = vdr.retrieve(queries, k=10, topk=768)

# 3. 处理结果
for i, query in enumerate(queries):
    print(f"\n查询: {query}")
    print("Top 10结果:")
    
    for j, (doc_id, score) in enumerate(zip(results.indices[i], results.scores[i])):
        doc = index.get_document(doc_id)
        print(f"  {j+1}. [{score:.2f}] {doc['title']}")
        print(f"     {doc['text'][:100]}...")
    
    # 可解释性分析
    print("\n语义解缠:")
    tokens = vdr.encoder_q.disentangle(query, topk=10)
    for token, weight in list(tokens.items())[:10]:
        print(f"  {token}: {weight:.2f}")
```

---

**文档版本**: v1.0  
**最后更新**: 2026-01-13  
**作者**: 基于VDR源码和论文整理  
**许可**: MIT License

如有问题或建议，欢迎提Issue或PR！ 🚀
