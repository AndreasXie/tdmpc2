import torch
from torch.nn import Buffer
import torch
import torch.nn as nn
from typing import Any, Dict
import numpy as np

class RunningScale(torch.nn.Module):
	"""Running trimmed scale estimator."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.value = Buffer(torch.ones(1, dtype=torch.float32, device=torch.device(cfg.get('device', 'cuda:0'))))
		self._percentiles = Buffer(torch.tensor([5, 95], dtype=torch.float32, device=torch.device(cfg.get('device', 'cuda:0'))))

	def state_dict(self):
		return dict(value=self.value, percentiles=self._percentiles)

	def load_state_dict(self, state_dict):
		self.value.copy_(state_dict['value'])
		self._percentiles.copy_(state_dict['percentiles'])

	def _positions(self, x_shape):
		positions = self._percentiles * (x_shape-1) / 100
		floored = torch.floor(positions)
		ceiled = floored + 1
		ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
		weight_ceiled = positions-floored
		weight_floored = 1.0 - weight_ceiled
		return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

	def _percentile(self, x):
		x_dtype, x_shape = x.dtype, x.shape
		x = x.flatten(1, x.ndim-1)
		in_sorted = torch.sort(x, dim=0).values
		floored, ceiled, weight_floored, weight_ceiled = self._positions(x.shape[0])
		d0 = in_sorted[floored] * weight_floored
		d1 = in_sorted[ceiled] * weight_ceiled
		return (d0+d1).reshape(-1, *x_shape[1:]).to(x_dtype)

	def update(self, x):
		percentiles = self._percentile(x.detach())
		value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
		self.value.data.lerp_(value, self.cfg.tau)

	def forward(self, x, update=False):
		if update:
			self.update(x)
		return x / self.value

	def __repr__(self):
		return f'RunningScale(S: {self.value})'

class RunningMeanStd:
    """
    使用 torch.Tensor 维护观测的均值和方差，用于归一化。
    """

    def __init__(self, shape, dtype=torch.float32, device=torch.device('cpu'), epsilon=1e-8):
        """
        初始化 RunningMeanStd。

        Args:
            shape (tuple): 观测的形状。
            dtype (torch.dtype): 数据类型。
            device (torch.device): 设备（CPU 或 GPU）。
            epsilon (float): 小常数，防止除以零。
        """
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(1e-4, dtype=dtype, device=device)  # 避免最初始时分母为0

        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        """
        使用新的批量数据更新均值和方差。

        Args:
            x (torch.Tensor): 输入数据，形状为 (batch_size, *shape)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("输入数据必须是 torch.Tensor 类型")

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int):
        """
        根据批量数据的均值和方差更新整体均值和方差。

        Args:
            batch_mean (torch.Tensor): 批量数据的均值。
            batch_var (torch.Tensor): 批量数据的方差。
            batch_count (int): 批量数据的样本数量。
        """
        batch_count = torch.tensor(batch_count, dtype=self.mean.dtype, device=self.mean.device)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入数据进行归一化。

        Args:
            x (torch.Tensor): 输入数据，形状为 (*, ...)

        Returns:
            torch.Tensor: 归一化后的数据。
        """
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def to_device(self, device: torch.device):
        """
        将内部张量移动到指定设备。

        Args:
            device (torch.device): 目标设备。
        """
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)

    def __repr__(self):
        return (f"RunningMeanStd(mean={self.mean}, var={self.var}, "
                f"count={self.count}, epsilon={self.epsilon})")
