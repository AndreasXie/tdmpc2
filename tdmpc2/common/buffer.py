import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device(cfg.get('device', 'cuda:0'))
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		# storage_device = 'cpu'
		print(f'Using {storage_device} memory for storage.')
		self._storage_device = torch.device(storage_device)
		# return self._reserve_buffer(
		# 	LazyMemmapStorage(self._capacity, scratch_dir="./tmp")
		# )
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, task

	def _prepare_batch_atari(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "done", "task", strict=False).to(self._device, non_blocking=True)
		obs = (td.get('obs')).float().contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		done = td.get('done')[1:].unsqueeze(-1).contiguous()
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, done, task
	
	def add(self, td):
		"""Add an episode to the buffer."""
		if 'obs' in td and self.cfg.has_done:
			# 确保数据在 [0, 255] 范围内，并转换为 uint8
			td['obs'] = td['obs'].clamp(0, 255).to(torch.uint8)
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		if self.cfg.has_done and self.cfg.task_platform == 'atari':
			td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
			return self._prepare_batch_atari(td)
		else:
			td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
			return self._prepare_batch(td)
		
class Buffer_atari():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device(cfg.get('device', 'cuda:0'))
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
		)
		self._batch_size = cfg.batch_size * (self.cfg.horizon + self.cfg.n_step+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		# storage_device = 'cpu'
		print(f'Using {storage_device} memory for storage.')
		self._storage_device = torch.device(storage_device)
		# return self._reserve_buffer(
		# 	LazyMemmapStorage(self._capacity, scratch_dir="./tmp")
		# )
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def _prepare_batch_atari(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "done", "task", strict=False).to(self._device, non_blocking=True)
		obs = (td.get('obs')).float().contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		done = td.get('done')[1:].unsqueeze(-1)
		task = td.get('task', None)
		gamma = self.cfg.get('discount', 0.997)

		if task is not None:
			task = task[0].contiguous()
		
		if self.cfg.n_step_return == True:
			discounted_reward = torch.zeros_like(reward[:self.cfg.horizon])
			terminated = torch.zeros_like(done[:self.cfg.horizon], dtype=torch.float32, device=done.device)
			for i in range(self.cfg.horizon):
				for k in range(self.cfg.n_step):
					discounted_reward[i] += gamma**k * reward[i+k] * (1 - terminated[i])
					terminated[i] = torch.clip_(terminated[i] + done[i+k], max=1.)

			return obs[:self.cfg.horizon+1], action[:self.cfg.horizon], discounted_reward, done[:self.cfg.horizon], terminated, task
		else:
			return obs[:self.cfg.horizon+1], action[:self.cfg.horizon], reward[:self.cfg.horizon], done[:self.cfg.horizon], task

	def add(self, td):
		"""Add an episode to the buffer."""
		if 'obs' in td and self.cfg.has_done:
			# 确保数据在 [0, 255] 范围内，并转换为 uint8
			td['obs'] = td['obs'].clamp(0, 255).to(torch.uint8)
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon + self.cfg.n_step+1).permute(1, 0)
		return self._prepare_batch_atari(td)
