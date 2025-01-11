import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union, Optional, Any, Callable

def create_norm(norm_type: str, num_channels: int):
    """
    根据 norm_type 创建对应的归一化层。
    norm_type 支持: 'none', 'bn', 'ln', 'gn'.
    """
    if norm_type == 'none':
        return nn.Identity()
    elif norm_type == 'bn':
        return nn.BatchNorm2d(num_channels)
    elif norm_type == 'ln':
        # PyTorch LayerNorm 通常对最后几个维度做归一化，这里假设输入形状 (N, C, H, W)，
        # 我们可以对通道和空间做 LayerNorm，因此需要 (C, H, W) 大小。但要注意可行性。
        return nn.LayerNorm([num_channels,], elementwise_affine=True)
    elif norm_type == 'gn':
        # 这里固定 group_size=8，如不符合你的期望可自行调整
        # 也可以改为 GroupNorm(num_groups, num_channels)
        # 例如：num_groups = max(1, num_channels // 8)
        group_size = 8
        num_groups = max(1, num_channels // group_size)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

def init_weights_xavier_uniform(module: nn.Module):
    """
    用于将可学习参数初始化为 xavier_uniform。若 fixup_init，则最后一层权重会设为 0。
    这个函数可以在模型构造后进行 apply。
    """
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_weights_fixup_final(module: nn.Module):
    """
    若 fixup_init=True，需要对最后一个卷积进行 0 初始化。
    可以在 ResidualBlock 的构造函数中对 final conv 做这个处理。
    """
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ResidualBlock(nn.Module):
    """
    一个 Impala 风格的残差块：
      - conv -> ReLU -> norm -> dropout -> conv -> ReLU -> norm -> conv -> + skip
      - 如果 fixup_init=True，则最后一层 conv 的权重用 0 初始化
    """
    def __init__(
        self,
        in_channels: int,
        norm_type: str = 'none',
        dropout: float = 0.0,
        fixup_init: bool = False,
    ):
        super().__init__()
        self.fixup_init = fixup_init

        # 第一条支路: conv -> relu -> norm -> dropout
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm1 = create_norm(norm_type, in_channels)
        # 这里用 2D dropout；若想精确模拟 JAX broadcast_dims，可自行改写
        self.dropout1 = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # 第二条支路: conv -> relu -> norm -> (最后一个 conv)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm2 = create_norm(norm_type, in_channels)
        self.conv3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True
        )

        # 如果 fixup_init=True，需要对 self.conv3 的权重初始化为 0
        if fixup_init:
            init_weights_fixup_final(self.conv3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.mish(x)
        out = self.norm1(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = F.mish(out)
        out = self.norm2(out)
        out = self.conv2(out)

        out = F.mish(out)
        # 最后一层 conv3 不再做 Norm + ReLU
        out = self.conv3(out)

        return out + residual

class ResidualStage(nn.Module):
    """
    一个残差阶段：包括一次卷积(可选 max-pool) + 若干个 ResidualBlock。
    Attributes:
      dims: 卷积和残差块的通道数
      num_blocks: 残差块数量
      use_max_pooling: 是否在进入残差块之前使用一次 3x3 stride=2 的 max-pool
      norm_type: 归一化方式
      fixup_init: 是否在最后一层 conv 初始化为 0
      dropout: dropout 概率
    """
    def __init__(
        self,
        dims: int,
        num_blocks: int,
        use_max_pooling: bool = True,
        norm_type: str = 'none',
        fixup_init: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_max_pooling = use_max_pooling
        self.num_blocks = num_blocks
        self.dims = dims

        # stage 入口的 conv
        self.conv_in = nn.Conv2d(
            in_channels=dims,
            out_channels=dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 池化层
        if use_max_pooling:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.Identity()

        # 构建多个残差块
        blocks = []
        for _ in range(num_blocks):
            block = ResidualBlock(
                in_channels=dims,
                norm_type=norm_type,
                fixup_init=fixup_init,
                dropout=dropout,
            )
            blocks.append(block)
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.pool(x)
        x = self.resblocks(x)
        return x


class ImpalaCNN(nn.Module):
    """
    一个 PyTorch 版本的 Impala CNN，实现类似 JAX 的结构与超参数控制。

    Args:
      width_scale: 宽度缩放因子
      dims: 每个 stage 的输出通道列表
      num_blocks: 每个 stage 内的残差块数量
      norm_type: 归一化模式: 'none', 'bn', 'ln', 'gn'
      fixup_init: 如果为 True，则各 ResidualBlock 最后一层 conv 用权重 0 初始化
      dropout: Dropout 概率
      in_channels: 输入图像的通道数 (如 RGB=3)
      use_init_xavier: 若为 True，则对卷积权重做 xavier_uniform 初始化
      use_max_pooling: Impala 原论文里，每个 stage 都有一次池化，可开关控制
    """
    def __init__(
        self,
        width_scale: float = 1.0,
        dims: Tuple[int, ...] = (16, 32, 32),
        num_blocks: int = 2,
        norm_type: str = 'none',
        fixup_init: bool = False,
        dropout: float = 0.0,
        in_channels: int = 4,
        use_init_xavier: bool = True,
        use_max_pooling: bool = True,
    ):
        super().__init__()
        self.width_scale = width_scale
        self.dims = dims
        self.num_blocks = num_blocks
        self.norm_type = norm_type
        self.fixup_init = fixup_init
        self.dropout = dropout
        self.in_channels = in_channels
        self.use_init_xavier = use_init_xavier
        self.use_max_pooling = use_max_pooling

        # 最开始：把输入通道调整到 dims[0] (如果 in_channels != dims[0])
        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(dims[0] * width_scale),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 按 dims 构建多个 stage
        stages = []
        prev_dim = int(dims[0] * width_scale)
        for width in dims[1:]:
            ch = int(width * width_scale)
            stage = ResidualStage(
                dims=ch,
                num_blocks=self.num_blocks,
                use_max_pooling=self.use_max_pooling,
                norm_type=self.norm_type,
                fixup_init=self.fixup_init,
                dropout=self.dropout,
            )
            # 先把通道变为 ch
            stages.append(
                nn.Conv2d(prev_dim, ch, kernel_size=3, stride=1, padding=1, bias=True)
            )
            stages.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                          if self.use_max_pooling else nn.Identity())
            # 再加上 stage
            stages.append(stage)
            prev_dim = ch
        self.stages = nn.Sequential(*stages)

        # 可选地初始化
        if self.use_init_xavier:
            self.apply(init_weights_xavier_uniform)
            # fixup_init=True 时，会在 ResidualBlock 构造时对 final conv 做 0 init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 先用 conv0
        x = self.conv0(x)

        # 2) 进入后续 stages
        x = self.stages(x)

        # 3) 最后做一次 ReLU
        x = F.mish(x)
        return x
    
def renormalize(tensor, has_batch=False):
    """
    重新归一化张量，使其值范围缩放到 [0, 1]。

    Args:
        tensor (torch.Tensor): 输入张量。
        has_batch (bool): 是否包含批量维度。如果为 False，则会在第0维扩展一个维度。

    Returns:
        torch.Tensor: 重新归一化后的张量，形状与输入相同。
    """
    shape = tensor.shape
    if not has_batch:
        tensor = tensor.unsqueeze(0)  # 在第0维扩展一个维度
    # 将张量展平为 (batch_size, -1)
    tensor_flat = tensor.view(tensor.size(0), -1)
    max_value, _ = tensor_flat.max(dim=1, keepdim=True)
    min_value, _ = tensor_flat.min(dim=1, keepdim=True)
    # 避免除以零，添加一个小的常数
    tensor_norm = (tensor_flat - min_value) / (max_value - min_value + 1e-5)
    # 将张量恢复到原始形状
    tensor_norm = tensor_norm.view(*shape)
    return tensor_norm

class ConvTMCell(nn.Module):
    """
    MuZero 风格的 SPR（Stochastic Prediction Representation）转换模型。

    Args:
        num_actions (int): 动作的数量，用于 one-hot 编码。
        in_channels (int): 输入张量的通道数。
        latent_dim (int): 潜在空间的维度，即卷积层的输出通道数。
        renormalize (bool): 是否对输出进行重新归一化。
        dtype (torch.dtype, 可选): 数据类型，默认为 torch.float32。
        initializer (callable, 可选): 卷积层权重初始化函数，默认为 Xavier 均匀初始化。
    """
    def __init__(self, num_actions, in_channels, latent_dim, renormalize=True, dtype=torch.float32, initializer=nn.init.xavier_uniform_):
        super(ConvTMCell, self).__init__()
        self.num_actions = num_actions
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.renormalize = renormalize
        self.dtype = dtype
        self.initializer = initializer

        # 定义卷积层
        # conv1 的输入通道数为 in_channels + num_actions
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels + self.num_actions,
            out_channels=self.latent_dim,
            kernel_size=3,
            stride=1,
            padding=1,  # 保持空间维度不变
            bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.latent_dim,
            out_channels=self.latent_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # 初始化卷积层权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化卷积层的权重。
        """
        self.initializer(self.conv1.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        self.initializer(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x, action):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
            action (torch.Tensor): 动作索引，形状为 (batch_size,)。
            eval_mode (bool, 可选): 是否处于评估模式。
            key (torch.Tensor, 可选): 随机数生成器种子（未使用）。

        Returns:
            tuple: (输出张量, 输出张量)。
        """
        batch_size, channels, height, width = x.shape

        # 将动作转换为 one-hot 编码
        # 扩展为与 x 相同的空间维度
        action_onehot = action.view(batch_size, self.num_actions, 1, 1)
        action_onehot = action_onehot.expand(-1, -1, height, width)  # (batch_size, num_actions, height, width)

        # 连接动作编码到输入张量的通道维度
        x = torch.cat([x, action_onehot], dim=1)  # 新的通道数为 in_channels + num_actions

        # 第一个卷积层
        x = self.conv1(x)
        x = F.relu(x)

        # 第二个卷积层
        x = self.conv2(x)
        x = F.relu(x)

        # 重新归一化（如果需要）
        if self.renormalize:
            x = renormalize(x, has_batch=True)

        return x
    
def test_ImpalaCNN():
    """
    测试 ImpalaCNN 类，确保其能够处理输入形状为 (4, 84, 84, 84) 的张量。
    """
    # 参数定义
    width_scale = 1.0
    dims = (16, 32, 32)  # 输出通道数，每个 stage 的通道数
    num_blocks = 2
    norm_type = 'none'  # 可选 'none', 'bn', 'ln', 'gn'
    fixup_init = False
    dropout = 0.0
    in_channels = 4  # 与输入张量的通道数一致
    use_init_xavier = True
    use_max_pooling = True

    # 创建 ImpalaCNN 实例
    model = ImpalaCNN(
        width_scale=width_scale,
        dims=dims,
        num_blocks=num_blocks,
        norm_type=norm_type,
        fixup_init=fixup_init,
        dropout=dropout,
        in_channels=in_channels,
        use_init_xavier=use_init_xavier,
        use_max_pooling=use_max_pooling,
    )

    # 打印模型结构（可选）
    print("ImpalaCNN 模型结构:")
    print(model)

    # 创建随机输入张量，形状为 (4, 84, 84, 84)
    batch_size = 4
    channels = 4
    height = 84
    width = 84
    x = torch.randn(batch_size, channels, height, width)

    # 前向传播
    output = model(x)

    # 打印输入和输出形状
    print(f"输入形状: {x.shape}")       # 预期: torch.Size([4, 84, 84, 84])
    print(f"输出形状: {output.shape}") # 预期: 根据模型结构，可能小于输入尺寸

def test_ConvTMCell():
    """
    测试 ConvTMCell 类，确保其能够处理输入形状为 (4, 84, 84, 84) 的张量和动作张量。
    """
    # 参数定义
    num_actions = 10
    in_channels = 32  # 与输入张量的通道数一致
    latent_dim = 32   # 保持输入和输出通道数一致
    renormalize_flag = True
    dtype = torch.float32
    initializer = nn.init.xavier_uniform_

    # 创建 ConvTMCell 实例
    model = ConvTMCell(
        num_actions=num_actions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        renormalize=renormalize_flag,
        dtype=dtype,
        initializer=initializer
    )

    # 打印模型结构（可选）
    print("ConvTMCell 模型结构:")
    print(model)

    # 创建随机输入张量，形状为 (4, 84, 84, 84)
    batch_size = 4
    channels = 32
    height = 6
    width = 6
    x = torch.randn(batch_size, channels, height, width)

    # 创建动作张量，形状为 (4,)
    actions = torch.randint(0, num_actions, (batch_size,))
    action_onehot = F.one_hot(actions, num_classes=num_actions).float()  # (batch_size, num_actions)

    # 前向传播
    output, _ = model(x, action_onehot.view(batch_size, num_actions, 1, 1))

    # 打印输入和输出形状
    print(f"输入形状: {x.shape}")       # 预期: torch.Size([4, 84, 84, 84])
    print(f"动作形状: {actions.shape}") # 预期: torch.Size([4])
    print(f"输出形状: {output.shape}") # 预期: torch.Size([4, 84, 84, 84])

if __name__ == "__main__":
    print("=== 测试 ImpalaCNN ===")
    test_ImpalaCNN()
    print("\n=== 测试 ConvTMCell ===")
    test_ConvTMCell()