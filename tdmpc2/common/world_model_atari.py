from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tensordict.nn import TensorDictParams

from common import layers, math, init


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		if not cfg.simba:
			self._encoder = layers.enc(cfg)
			self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
			self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, [cfg.mlp_dim], 1)

			self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
			self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.action_dim)
			self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		else:
			self._encoder = layers.enc(cfg)
			self._dynamics = layers.res_mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
			self._termination = layers.res_mlp(cfg.latent_dim + cfg.task_dim, [cfg.mlp_dim], 1)

			self._reward = layers.res_mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
			self._pi = layers.res_mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.action_dim)
			self._Qs = layers.Ensemble([layers.res_mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"] if not cfg.simba else self._Qs.params["1", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		self._detach_Qs.params = self._detach_Qs_params
		self._target_Qs.params = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Termination', 'Reward', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._termination, self._reward, self._pi, self._Qs]):
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)

		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)
	
	def terminated(self, z, task):
		"""
		Predicts the termination probability given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return torch.sigmoid(self._termination(z))

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def _discrete_pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		logits = self._pi(z)
		policy_dist = Categorical(logits=logits)
		a1 = policy_dist.sample()

		if len(a1.shape)==2: 
			actions = torch.reshape(a1, (a1.shape[0], a1.shape[1], 1))
		elif len(a1.shape)==1: 
			actions = a1
		action_probs = policy_dist.probs
		log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

		onehot_action = math.int_to_one_hot(actions, self.cfg.action_dim)

		return actions, onehot_action, action_probs, log_probs

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		Policy can be either continuous (Gaussian) or discrete (categorical).
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		return self._discrete_pi(z, task)

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2

	def reset(self, cfg):
		"""
		Reset target Q-networks.
		"""
		old_encoder = deepcopy(self._encoder).to("cpu")
		old_dynamics = deepcopy(self._dynamics).to("cpu")
		self._encoder = layers.enc(cfg)
		if cfg.simba:
			self._dynamics = layers.res_mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
			self._termination = layers.res_mlp(cfg.latent_dim + cfg.task_dim, [cfg.mlp_dim], 1)
			self._reward = layers.res_mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
			self._pi = layers.res_mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim if cfg.action == 'continuous' else cfg.action_dim)
			self._Qs = layers.Ensemble([layers.res_mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		else:
			self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
			self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, [cfg.mlp_dim], 1)
			self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
			self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim if cfg.action == 'continuous' else cfg.action_dim)
			self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		
		self.apply(init.weight_init)

		layer_num = len(self._encoder)
		for key in self._encoder.keys():
			for layer in range(layer_num):
				new_layer_dict = self._encoder[key][layer].state_dict()
				old_layer_dict = old_encoder[key][layer].state_dict()
				for key in new_layer_dict.keys():
					if "weight" in key:
						new_layer_dict[key] = (1.0 - cfg.reset_percent) * old_layer_dict[key] + cfg.reset_percent * new_layer_dict[key]
					elif "bias" in key:
						new_layer_dict[key] = new_layer_dict[key]
		#reset dynamic
		for name, param in self._dynamics.named_parameters():
			if param.requires_grad:
				# 获取旧的参数
				old_param = old_dynamics.state_dict().get(name)
				if old_param is not None:
					# 确保 old_param 和 param.data 在同一设备上
					old_param = old_param.to(param.device)
					# 线性插值: new_param = (1 - reset_percent) * old_param + reset_percent * new_param
					param.data = (1.0 - cfg.reset_percent) * old_param + cfg.reset_percent * param.data


		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
		self.init()