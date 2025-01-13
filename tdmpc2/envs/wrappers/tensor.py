from collections import defaultdict

import gym
import numpy as np
import torch
import common.math as math
import gymnasium
class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env, cfg=None):
		super().__init__(env)
		self.action_mode = cfg.get('action', 'multistep_randomshooting')
		self.action_space = env.action_space
	
	def rand_act(self):
		if self.action_mode == 'category':
			#generate random distribution over actions number
			act_probs = np.random.rand(self.action_space.n)
			return torch.tensor(act_probs.astype(np.float32))
		elif self.action_mode == 'multistep_randomshooting':
			action = torch.tensor(self.action_space.sample(), dtype=torch.int64)
			return math.int_to_one_hot(action, self.action_space.n)

		sample = [self.action_space.sample()]
		sample = np.array(sample)
		# Convert to tensor
		if sample.ndim == 2:
			sample = sample[0]

		return torch.from_numpy(sample.astype(np.float32))

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x)
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		elif isinstance(obs,tuple):
			obs = self._try_f32_tensor(obs[0])
		else:
			obs = self._try_f32_tensor(obs)

		return obs

	def reset(self, task_idx=None):
		return self._obs_to_tensor(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), torch.tensor(done, dtype=torch.float32), info

class TensorWrapperAtari(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env, cfg=None):
		super().__init__(env)
		self.action_space = env.action_space
	
	def rand_act(self):
		action = torch.tensor(self.action_space.sample(), dtype=torch.int64)
		return math.int_to_one_hot(action, self.action_space.n)

	def _try_f32_tensor(self, x):
		x = torch.Tensor(x)
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		elif isinstance(obs,tuple):
			obs = self._try_f32_tensor(obs[0])
		else:
			obs = self._try_f32_tensor(obs)

		return obs

	def reset(self, task_idx=None):
		return self._obs_to_tensor(self.env.reset())

	def step(self, action):
		obs, reward, done, trunc, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), torch.tensor(done, dtype=torch.float32), trunc, info

	def render(self, mode='rgb_array'):
		return self.env.render(mode)
	
	def save_video(self):
		self.env.save_video()