import math
from torch.distributions import TransformedDistribution, Normal, constraints
from torch.distributions import transforms
import torch.nn.functional as F
import numpy as np

# 定义 TanhTransform
class TanhTransform(transforms.Transform):
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

# 定义 SquashedNormal 分布
class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        transforms_ = [TanhTransform()]
        super().__init__(self.base_dist, transforms_)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

# 定义 MinMaxStats 类
class MinMaxStats:
    def __init__(self, minmax_delta, min_value_bound=None, max_value_bound=None):
        """
        Minimum and Maximum statistics
        :param minmax_delta: float, for soft update
        :param min_value_bound:
        :param max_value_bound:
        """
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')
        self.minmax_delta = minmax_delta

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            if value >= self.maximum:
                value = self.maximum
            elif value <= self.minimum:
                value = self.minimum
            # 仅在设置了最大值和最小值时进行归一化
            value = (value - self.minimum) / max(self.maximum - self.minimum, self.minmax_delta)  # [0, 1] 范围

        value = max(min(value, 1), 0)
        return value

    def clear(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

# 定义 softmax 函数
def softmax(logits):
    # logits = np.asarray(logits)
    logits -= logits.max()
    e_x = np.exp(logits)
    return e_x / e_x.sum()