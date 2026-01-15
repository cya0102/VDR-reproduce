"""VDR跨模态图像编码器模块

该模块实现了VDR（Vision Document Retrieval）的图像编码器，
将图像编码为稀疏的词袋表示，用于跨模态检索任务。
"""

import logging
from collections import OrderedDict
from contextlib import nullcontext
from typing import List, Union

from PIL import Image
import torch
from torch import Tensor as T
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from wordcloud import WordCloud

from .vdr_crossmodal_text import VALID_TOKEN_IDS, VID2LID
from ..utils.sparsify_utils import build_bow_mask, build_topk_mask, elu1p
from ..utils.visualize_utils import wordcloud_from_dict

logger = logging.getLogger(__name__)


class Bottleneck(torch.torch.nn.Module):
    """ResNet的瓶颈块（Bottleneck Block）
    
    实现了ResNet中的瓶颈结构，使用1x1、3x3、1x1的卷积序列，
    可以有效减少参数量并增加网络深度。
    
    Attributes:
        expansion: 通道扩展倍数，固定为4
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """初始化瓶颈块
        
        Args:
            inplanes: 输入通道数
            planes: 中间层通道数
            stride: 步长，大于1时会进行下采样
        """
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)

        self.conv2 = torch.nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.avgpool = torch.nn.AvgPool2d(stride) if stride > 1 else torch.nn.Identity()

        self.conv3 = torch.nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = torch.nn.Sequential(OrderedDict([
                ("-1", torch.nn.AvgPool2d(stride)),
                ("0", torch.nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", torch.nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            经过瓶颈块处理后的输出张量
        """
        identity = x

        # 1x1卷积降维
        out = self.relu(self.bn1(self.conv1(x)))
        # 3x3卷积提取特征
        out = self.relu(self.bn2(self.conv2(out)))
        # 如果stride>1，进行平均池化下采样
        out = self.avgpool(out)
        # 1x1卷积升维
        out = self.bn3(self.conv3(out))

        # 如果需要下采样，对恒等映射也进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(torch.nn.Module):
    """2D注意力池化层
    
    使用多头注意力机制对2D特征图进行全局池化，
    相比普通的平均池化能更好地聚合空间信息。
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """初始化注意力池化层
        
        Args:
            spacial_dim: 空间维度（特征图的宽或高）
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            output_dim: 输出维度，默认与embed_dim相同
        """
        super().__init__()
        self.positional_embedding = torch.nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.c_proj = torch.nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征图，形状为[N, C, H, W]
            
        Returns:
            池化后的特征向量，形状为[N, output_dim]
        """
        # 将特征图reshape为序列形式: NCHW -> (HW)NC
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        # 添加全局平均池化作为第一个token: (HW+1)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        # 添加位置编码
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class LayerNorm(torch.nn.LayerNorm):
    """层归一化（Layer Normalization）
    
    继承自torch的LayerNorm，增加了对fp16（半精度浮点数）的支持。
    在计算时转换为fp32以保持数值稳定性，输出时恢复原始数据类型。
    """

    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            归一化后的张量，保持原始数据类型
        """
        orig_type = x.dtype
        # 转换为fp32进行计算，保证数值稳定性
        ret = super().forward(x.type(torch.float32))
        # 转换回原始数据类型
        return ret.type(orig_type)

class QuickGELU(torch.nn.Module):
    """快速GELU激活函数
    
    GELU的近似实现，使用sigmoid函数加速计算。
    相比标准GELU，计算效率更高但精度略有损失。
    """
    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            激活后的张量
        """
        # GELU的快速近似: x * sigmoid(1.702 * x)
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(torch.nn.Module):
    """残差注意力块
    
    实现了Transformer中的一个标准层，包含多头自注意力和前馈网络，
    两者都使用了残差连接和层归一化。
    """
    def __init__(self, d_model: int, n_head: int):
        """初始化残差注意力块
        
        Args:
            d_model: 模型维度
            n_head: 注意力头数
        """
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", torch.nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """应用多头自注意力机制
        
        Args:
            x: 输入张量
            attn_mask: 注意力掩码，用于屏蔽某些位置
            
        Returns:
            注意力输出
        """
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask=None):
        """前向传播
        
        Args:
            x: 输入张量
            attn_mask: 注意力掩码
            
        Returns:
            输出张量
        """
        # 多头自注意力 + 残差连接
        x = x + self.attention(self.ln_1(x), attn_mask)
        # 前馈网络 + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(torch.nn.Module):
    """Transformer编码器
    
    由多个残差注意力块堆叠而成，用于提取序列特征。
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """初始化Transformer
        
        Args:
            width: 模型宽度（隐藏层维度）
            layers: 层数
            heads: 注意力头数
            attn_mask: 注意力掩码
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.attn_mask = attn_mask
        self.heads = heads
        self.resblocks = torch.nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask=None):
        """前向传播
        
        Args:
            x: 输入张量，形状为[L, N, D]
            attn_mask: 注意力掩码
            
        Returns:
            编码后的张量，形状为[L, N, D]
        """
        if attn_mask is not None:
            # 将文本的二值掩码转换为3D注意力掩码
            attn_mask = (1.0 - attn_mask) * -10000.0  # NL，将0变为-10000，1变为0
            attn_mask = attn_mask.unsqueeze(1)  # N 1 L
            target_seq_length = attn_mask.shape[-1]
            # 为每个注意力头复制掩码: N*heads, L, L
            attn_mask = attn_mask.repeat(self.heads, target_seq_length, 1)
        # 逐层处理
        for i, layer_module in enumerate(self.resblocks):
            x = layer_module(x, attn_mask)

        return x



class VDRImageEncoderConfig(PretrainedConfig):
    """VDR图像编码器配置类
    
    定义了图像编码器的所有超参数。
    """
    def __init__(
        self,
        resolution=224,
        tokenizer_id="bert-base-uncased",
        patch_size=32,
        width=768,
        layers=12,
        heads=12, 
        topk=768,
        **kwargs,
    ):
        """初始化配置
        
        Args:
            resolution: 图像分辨率
            tokenizer_id: 分词器ID，用于词汇表映射
            patch_size: 图像分块大小
            width: Transformer宽度（隐藏层维度）
            layers: Transformer层数
            heads: 注意力头数
            topk: 保留的top-k稀疏维度数
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        self.tokenizer_id = tokenizer_id
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.topk = topk


class VDRImageEncoder(PreTrainedModel):
    """VDR图像编码器
    
    将图像编码为稀疏的词袋表示（Bag-of-Words），用于跨模态检索。
    核心思想是将图像的视觉特征映射到文本词汇空间，实现视觉-语言的统一表示。
    
    主要流程：
    1. 通过卷积将图像分块并投影到嵌入空间
    2. 添加位置编码
    3. 通过Transformer提取特征
    4. 投影到词汇表空间
    5. 进行稀疏化处理（top-k选择）
    """
    config_class = VDRImageEncoderConfig

    def __init__(self, config: VDRImageEncoderConfig, **kwargs):
        """初始化图像编码器
        
        Args:
            config: 编码器配置
        """
        super().__init__(config, **kwargs)
        self.config = config
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=config.width, kernel_size=config.patch_size, stride=config.patch_size, bias=False)
        scale = config.width ** -0.5
        self.positional_embedding = torch.nn.Parameter(scale * torch.randn((config.resolution // config.patch_size) ** 2, config.width))
        self.ln_pre = LayerNorm(config.width)
        self.transformer = Transformer(config.width, config.layers, config.heads)
        self.ln_post = LayerNorm(config.width)
        self.proj = torch.nn.Parameter(torch.ones([len(VALID_TOKEN_IDS), config.width]))
        self.valid_token_ids = VALID_TOKEN_IDS
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)

    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: 输入图像张量，形状为[N, 3, resolution, resolution]
            
        Returns:
            图像特征，形状为[N, num_patches, width]
        """
        # 卷积分块：将图像分割成patches并投影到嵌入空间
        x = self.conv1(x)  # shape = [N, width, grid, grid]
        # 将2D特征图展平为序列
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [N, width, grid^2]
        x = x.permute(0, 2, 1)  # shape = [N, grid^2, width]
        # 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)
        # 预层归一化
        x = self.ln_pre(x)
        # 调整维度顺序以适配Transformer: [N, L, D] -> [L, N, D]
        x = x.permute(1, 0, 2)
        # 通过Transformer提取特征
        x = self.transformer(x)
        # 恢复维度顺序: [L, N, D] -> [N, L, D]
        x = x.permute(1, 0, 2)
        # 后层归一化
        x = self.ln_post(x)
        return x

    def embed(self, images: Union[str, List[str], T], training: bool = False, topk=None):
        """将图像编码为稀疏词袋表示
        
        完整的编码流程，包括特征提取、投影、池化、激活、归一化和稀疏化。
        
        Args:
            images: 输入图像，可以是：
                - 单个图像路径（str）
                - 图像路径列表（List[str]）
                - 图像张量（Tensor）
            training: 是否为训练模式，False时会禁用梯度计算
            topk: 保留的top-k维度数，默认使用config中的值
            
        Returns:
            稀疏词袋表示，形状为[N, vocab_size]
        """
        topk = topk or self.config.topk
        # 训练模式下需要梯度，推理模式下禁用梯度以节省内存
        with torch.no_grad() if not training else nullcontext():
            # 处理输入：统一转换为张量格式
            if isinstance(images, str):
                images = [images]
            if isinstance(images, List) and isinstance(images[0], str):
                images = [self.load_image_file(image) for image in images]
                images = torch.cat(images, dim=0)
            # 提取图像特征
            img_emb = self(images.type(self.dtype).to(self.device))  # [N, L, D]
            # 投影到词汇表空间
            img_emb = img_emb @ self.proj.t()  # [N, L, V]
            # 在patch维度上取最大值进行池化
            img_emb = img_emb.max(1)[0]  # [N, V]
            # 使用elu1p激活函数（保证非负）
            img_emb = elu1p(img_emb)
            # L2归一化
            img_emb = F.normalize(img_emb)
            # 构建top-k掩码，保留最重要的k个维度
            topk_mask = build_topk_mask(img_emb, k=topk)
            # 应用掩码，得到稀疏表示
            img_emb = img_emb * topk_mask
        return img_emb

    def load_image_file(self, file_path):
        """加载并预处理图像文件
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            预处理后的图像张量，形状为[1, 3, resolution, resolution]
        """
        # 读取图像并转换为RGB格式
        image = Image.open(file_path).convert('RGB')
        # 应用预处理（resize、crop、归一化等）
        image = preprocess(image)
        # 添加batch维度
        return image.unsqueeze(0)

    def disentangle(self, image: Union[str, T], topk: int = None, visual=False, save_file=None):
        """解析图像的语义表示
        
        将图像编码为词袋表示后，提取top-k个最重要的词及其权重，
        用于可解释性分析和可视化。
        
        Args:
            image: 输入图像（路径或张量）
            topk: 返回的top-k词数量
            visual: 是否生成词云可视化
            save_file: 词云保存路径（仅当visual=True时有效）
            
        Returns:
            字典，键为词token，值为对应的权重
        """
        topk = topk or self.config.topk
        # 获取图像的稀疏表示并提取top-k
        topk_result = self.embed(image).topk(topk)
        # 获取top-k的索引
        topk_token_ids = topk_result.indices.flatten().tolist()
        # 将词汇表索引映射到实际的token ID
        topk_token_ids = [VID2LID[x] for x in topk_token_ids]
        # 获取对应的权重值
        topk_values = topk_result.values.flatten().tolist()
        # 将token ID转换为实际的词
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_token_ids)
        # 构建词-权重字典
        results = dict(zip(topk_tokens, topk_values))
        # 如果需要，生成词云可视化
        if visual:
            wordcloud_from_dict(results, max_words=topk, save_file=save_file)
        return results


    # disentangle方法的简写别名
    dst = disentangle

    def display_image(self, image: Union[str, T] = None, save_file=None):
        """显示和保存图像
        
        对图像进行预处理（中心裁剪为正方形），并可选保存。
        
        Args:
            image: 输入图像（路径或PIL Image对象）
            save_file: 保存路径，为None则不保存
            
        Returns:
            处理后的PIL Image对象
        """
        # 如果是路径，先加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        # 计算裁剪尺寸（取宽高的最小值，保证正方形）
        crop_size = min(image.width, image.height)
        # 构建预处理pipeline：resize到正方形，然后中心裁剪
        preprocess_ = Compose([
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
        ])
        image = preprocess_(image)
        # 保存图像（如果指定了保存路径）
        if save_file is not None:
            image.save(save_file, format='PNG')
        return image
        



# 用于嵌入编码的图像预处理pipeline
# 包括resize、中心裁剪、转张量、标准化（使用ImageNet的均值和标准差）
preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),  # 缩放到224x224
    CenterCrop(224),  # 中心裁剪
    ToTensor(),  # 转换为张量，值域[0,1]
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # ImageNet标准化
])

# 用于可视化的图像预处理pipeline
# 不包括归一化，以便直接显示
preprocess_ = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
])
