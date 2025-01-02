import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from common import math, init, layers
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict
from mcts.mcts_muzero import PyMCTS

class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device(cfg.get('device', 'cuda:0'))
		self.model = WorldModel(cfg).to(self.device)
		if cfg.optimizer == 'adam':
			self.optim = torch.optim.AdamW([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				 }
			], lr=self.cfg.lr, capturable=True)
			self.pi_optim = torch.optim.AdamW(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

		elif cfg.optimizer == 'sgd':
			self.optim = torch.optim.SGD([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				 }
			], lr=self.cfg.lr, weight_decay=0.0001,momentum=0.9)
			self.pi_optim = torch.optim.SGD(self.model._pi.parameters(), lr=self.cfg.lr, weight_decay=0.0001,momentum=0.9)

		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
		if cfg.get('action') == 'mcts':
			self.step_counter = 0
			self.mcts_temperature = cfg.mcts_temperature
			self.mcts = PyMCTS(self.cfg)

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan_multistep_randomshooting if self.cfg.action == 'multistep_randomshooting' else self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max) if self.cfg.get('task_platform') != 'atari' else self.cfg.get('discount', 0.997)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		prob_entropy = 0
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			action, prob_entropy = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
			prob_entropy = prob_entropy.mean()
		else:
			z = self.model.encode(obs, task)
			action = self.model.pi(z, task)[1]
			if self.cfg.get('task_platform') == 'atari':
				action = action.squeeze(0) # TODO: this is a bit hacky
		return action.cpu(), prob_entropy

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		pi = self.model.pi(z, task)[1]
		if self.cfg.action == 'discrete':
			pi = pi.squeeze(1) # TODO: this is a bit hacky
		return G + discount * self.model.Q(z, pi, task, return_type='avg')

	# @torch.no_grad()
	# def _plan_multistep_randomshooting(self, obs, t0=False, eval_mode=False, task=None):
	# 	"""
	# 	Plan a sequence of actions using multi-step MPPI for a discrete action space.
	# 	"""

	# 	# Encode observation
	# 	z = self.model.encode(obs, task)

	# 	# Compute pi_actions if using policy trajectories
	# 	if self.cfg.num_pi_trajs > 0:
	# 		pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
	# 		_z = z.repeat(self.cfg.num_pi_trajs, 1)
	# 		for t in range(self.cfg.horizon-1):
	# 			action = self.model.pi(_z, task)[1]
	# 			if self.cfg.action == 'discrete':
	# 				action = action.squeeze(1)  # Ensure shape: (num_pi_trajs, action_dim)
	# 			pi_actions[t] = action
	# 			_z = self.model.next(_z, pi_actions[t], task)
	# 		action = self.model.pi(_z, task)[1]
	# 		if self.cfg.action == 'discrete':
	# 			action = action.squeeze(1)
	# 		pi_actions[-1] = action

	# 	# Repeat z for all samples
	# 	z = z.repeat(self.cfg.num_samples, 1)

	# 	# Initialize probability distribution over actions (T,A)
	# 	if t0:
	# 		# If first time step in episode, start with uniform distribution
	# 		prob_action = torch.full((self.cfg.horizon, self.cfg.action_dim), 1.0 / self.cfg.action_dim, device=self.device)
	# 	else:
	# 		# Otherwise start from previous distribution if available, or uniform if not
	# 		if hasattr(self, '_prev_prob'):
	# 			prob_action = self._prev_prob.clone()
	# 		else:
	# 			prob_action = torch.full((self.cfg.horizon, self.cfg.action_dim), 1.0 / self.cfg.action_dim, device=self.device)

	# 	# Multi-step MPPI iterations
	# 	for _ in range(self.cfg.iterations):
	# 		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)

	# 		# Fill in policy trajectories if any
	# 		if self.cfg.num_pi_trajs > 0:
	# 			actions[:, :self.cfg.num_pi_trajs] = pi_actions

	# 		# Sample from current prob_action
	# 		# Categorical expects (batch, action_dim), here batch = horizon, so it returns T distributions
	# 		# Sampling shape: (N - num_pi_trajs) samples from each of T distributions -> shape (N - num_pi_trajs, T)
	# 		dist = Categorical(prob_action)
	# 		actions_sample = dist.sample((self.cfg.num_samples - self.cfg.num_pi_trajs,))  # (N-num_pi, T)
	# 		actions_sample = actions_sample.t()  # (T, N-num_pi)
	# 		actions[:, self.cfg.num_pi_trajs:] = math.int_to_one_hot(actions_sample, self.cfg.action_dim)

	# 		# Optional: Apply task mask if multitask
	# 		if self.cfg.multitask:
	# 			actions = actions * self.model._action_masks[task]

	# 		# Evaluate trajectories
	# 		value = self._estimate_value(z, actions, task).nan_to_num(0)

	# 		# Select elite trajectories
	# 		elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
	# 		elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

	# 		# Compute weighted scores
	# 		max_value = elite_value.max(0).values
	# 		score = torch.exp(self.cfg.temperature * (elite_value - max_value))
	# 		score = score / (score.sum(0) + 1e-9)  # Normalize

	# 		# Update probability distribution over actions:
	# 		# Weighted sum over elites to get action counts
	# 		# elite_actions shape: (T, K, A) where K = num_elites
	# 		# score shape: (K, 1)
	# 		weighted_counts = (elite_actions * score.unsqueeze(0)).sum(dim=1)  # (T, A)

	# 		# Convert counts to probabilities
	# 		# Add epsilon to avoid log(0)
	# 		eps = 1e-6
	# 		new_prob = weighted_counts + eps
	# 		new_prob = new_prob / new_prob.sum(dim=-1, keepdim=True)

	# 		# Blend with old distribution for stability, alpha as a step size
	# 		alpha = self.cfg.mppi_alpha if hasattr(self.cfg, 'mppi_alpha') else 0.5
	# 		prob_action = (1 - alpha) * prob_action + alpha * new_prob

	# 	# After finishing all iterations, sample action from final distribution or return mean action
	# 	# Here we return the first action of the horizon.
	# 	# If eval_mode, pick argmax action; else sample stochastically:
	# 	if eval_mode:
	# 		a_idx = prob_action[0].argmax(dim=-1)
	# 	else:
	# 		final_dist = Categorical(prob_action[0].unsqueeze(0)) # shape (1, A)
	# 		a_idx = final_dist.sample().squeeze(0)

	# 	a = math.int_to_one_hot(a_idx, self.cfg.action_dim)
	# 	# Store prob_action for next step if desired
	# 	if not hasattr(self, '_prev_prob'):
	# 		self._prev_prob = prob_action.clone()
	# 	else:
	# 		self._prev_prob.copy_(prob_action)

	# 	return a

	@torch.no_grad()
	def _plan_multistep_randomshooting(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using multi-step MPPI for a discrete action space.
		"""
		# 1) Encode observation
		z = self.model.encode(obs, task)

		# 2) Optionally fill policy trajectories (if using num_pi_trajs)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				action = self.model.pi(_z, task)[1]
				if self.cfg.action == 'discrete':
					action = action.squeeze(1)  # (num_pi_trajs, action_dim)
				pi_actions[t] = action
				_z = self.model.next(_z, pi_actions[t], task)
			action = self.model.pi(_z, task)[1]
			if self.cfg.action == 'discrete':
				action = action.squeeze(1)
			pi_actions[-1] = action

		# 3) Repeat latent state for all samples
		z = z.repeat(self.cfg.num_samples, 1)

		# 4) Initialize probability distribution over actions (T,A)
		if t0:
			# first time step in episode -> uniform
			prob_action = torch.full((self.cfg.horizon, self.cfg.action_dim),
									1.0 / self.cfg.action_dim, device=self.device)
		else:
			# otherwise from previous distribution if available
			if hasattr(self, '_prev_prob'):
				prob_action = self._prev_prob.clone()
			else:
				prob_action = torch.full((self.cfg.horizon, self.cfg.action_dim),
										1.0 / self.cfg.action_dim, device=self.device)
				
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
					# (a) Fill in policy trajectories if any
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# 5) Multi-step MPPI / random-shooting iterations
		for iteration in range(self.cfg.iterations):
			# (b) Sample the rest from current prob_action
			dist = Categorical(prob_action)  
			actions_sample = dist.sample((self.cfg.num_samples - self.cfg.num_pi_trajs,))  # shape: (N-num_pi, T)
			actions_sample = actions_sample.t()  # shape: (T, N-num_pi)
			actions[:, self.cfg.num_pi_trajs:] = math.int_to_one_hot(actions_sample, self.cfg.action_dim)

			# (c) Optional: apply task mask
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# (d) Evaluate
			value = torch.exp(self._estimate_value(z, actions, task).nan_to_num(0) + 1e-6)

			importance_dist = (actions.permute(1,0,2) - prob_action).permute(1,0,2)

			weighted_value = value * importance_dist

			new_prob = weighted_value.sum(dim=1) / value.sum()

			# # (e) Compute weights based on all trajectories using min-max normalization
			# weights = self.cfg.temperature * (value - value.max())
			# normalized_weights = F.softmax(weights, dim=1)

			# # (f) Compute weighted counts
			# importance_dist = (actions.permute(1,0,2) - prob_action).permute(1,0,2)
			# importance_weights = importance_dist * normalized_weights + 1e-6
			# weighted_counts = (importance_weights).sum(dim=1)  # shape: (T, A)

			# # (g) Convert counts to probabilities
			# new_prob = weighted_counts + 1e-6
			# new_prob = new_prob / (normalized_weights.sum(dim=-1, keepdim=True))

			# (h) Blend with old distribution
			alpha = self.cfg.mppi_alpha if hasattr(self.cfg, 'mppi_alpha') else 0.5
			
			prob_action = prob_action + alpha * new_prob 

			# (i) Add Dirichlet noise for exploration
			# dirichlet_noise = torch.distributions.Dirichlet(torch.tensor([self.cfg.dirichlet_alpha]*new_prob.shape[-1]*self.cfg.horizon)).sample().to(self.device).reshape(self.cfg.horizon, self.cfg.action_dim)
			# prob_action= (1 - self.cfg.explore_frac) * prob_action+ self.cfg.explore_frac * dirichlet_noise

		# 6) After finishing all iterations, pick or sample the first action
		dist = Categorical(prob_action)  
		actions_sample = dist.sample((self.cfg.num_samples - self.cfg.num_pi_trajs,))  # shape: (N-num_pi, T)
		actions_sample = actions_sample.t()  # shape: (T, N-num_pi)
		actions[:, self.cfg.num_pi_trajs:] = math.int_to_one_hot(actions_sample, self.cfg.action_dim)

		# Compute elite actions
		value = self._estimate_value(z, actions, task).nan_to_num(0)
		elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
		elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

		# Sample action according to score
		max_value = elite_value.max(0).values
		score = torch.exp(self.cfg.temperature*(elite_value - max_value))
		score = score / score.sum(0)

		tau = 0.5 if eval_mode else 1.0
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1),temperature=tau)  # gumbel_softmax_sample is compatible with cuda graphs
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		entropy = dist.entropy()

		if not hasattr(self, '_prev_prob'):
			self._prev_prob = prob_action.clone()
		else:
			self._prev_prob.copy_(prob_action)

		return actions[0], entropy

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.action == 'mcts':
			self.step_counter += 1
			mcts = PyMCTS(self.cfg)
			training_steps = self.cfg.steps - self.cfg.seed_steps

			if self.step_counter == training_steps/2:
				self.mcts_temperature = 0.5 
			elif self.step_counter == training_steps*3/4:
				self.mcts_temperature = 0.25

			noise = not eval_mode
			_, best_actions, _ = mcts.search(self.model, obs.shape[0], z, task, 0, self.mcts_temperature, self.discount, noise, self.device)
			return math.int_to_one_hot(torch.Tensor(best_actions).long(),self.cfg.action_dim)

		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				action = self.model.pi(_z, task)[1]
				if self.cfg.action == 'discrete':
					action = action.squeeze(1)
				pi_actions[t] = action
				_z = self.model.next(_z, pi_actions[t], task)
			action = self.model.pi(_z, task)[1]
			if self.cfg.action == 'discrete':
				action = action.squeeze(1)
			pi_actions[-1] = action

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		if self.cfg.action == 'continuous':
			mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
			std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
			if not t0:
				mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Random shooting
		if self.cfg.action == 'discrete':
			# Sample actions
			actions_sample = torch.randint(0, self.cfg.action_dim, (self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs), device=actions.device)
			actions[:, self.cfg.num_pi_trajs:] = math.int_to_one_hot(actions_sample, self.cfg.action_dim)

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Sample action according to score
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
			actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
			entropy = Categorical(probs=score).entropy()
			return actions[0], entropy

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)

		return a.clamp(-1, 1)

	def update_pi(self, z, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		_, _, action_probs, log_probs = self.model.pi(z, task)
		actions = actions = torch.eye(self.cfg.action_dim, device=z.device).unsqueeze(0)
		if z.dim() == 2:
			# z (batch_size, latent_dim) -> (batch_size, action_dim, latent_dim)
			z = z.unsqueeze(1).expand(-1, self.cfg.action_dim, -1)
			actions = actions.repeat(z.shape[0], 1, 1)
		elif z.dim() == 3:
			# z (seq_len, batch_size, latent_dim) -> (seq_len, batch_size, action_dim, latent_dim)
			z = z.unsqueeze(2).expand(-1, -1, self.cfg.action_dim, -1)
			actions = actions.unsqueeze(0).repeat(z.shape[0], z.shape[1], 1, 1)
		
		qs = self.model.Q(z, actions, task, return_type='avg', detach=True).squeeze(-1)
		self.scale.update(torch.sum(action_probs*qs,dim=(1,2),keepdim=True)[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((action_probs*((self.cfg.entropy_coef * log_probs) - qs)).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)
		if self.cfg.autotune:
			alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_probs + self.target_entropy).detach())).mean()
			self.a_optimizer.zero_grad()
			alpha_loss.backward()
			self.a_optimizer.step()
			self.cfg.entropy_coef = self.log_alpha.exp().item()

		return pi_loss.detach(), pi_grad_norm
	
	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		_, _, next_act_prob, next_log_prob = self.model.pi(next_z, task)
		actions = torch.eye(self.cfg.action_dim, device=next_z.device).unsqueeze(0)
		if next_z.dim() == 2:
			# z (batch_size, latent_dim) -> (batch_size, action_dim, latent_dim)
			next_z = next_z.unsqueeze(1).expand(-1, self.cfg.action_dim, -1)
			actions = actions.repeat(next_z.shape[0], 1, 1)
		elif next_z.dim() == 3:
			# z (seq_len, batch_size, latent_dim) -> (seq_len, batch_size, action_dim, latent_dim)
			next_z = next_z.unsqueeze(2).expand(-1, -1, self.cfg.action_dim, -1)
			actions = actions.unsqueeze(0).repeat(next_z.shape[0], next_z.shape[1], 1, 1)
		# encoded_action = encoded_action.squeeze(2)
		qs = self.model.Q(next_z, actions, task, return_type='min', target=True).squeeze(3)

		min_q_next_target = next_act_prob * (qs - self.cfg.entropy_coef * next_log_prob)
		min_q_next_target = min_q_next_target.sum(dim=2, keepdim=True)

		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		td_targets = reward + discount * min_q_next_target
		return td_targets
	
	@torch.no_grad()
	def _td_target_term(self, next_z, reward, done, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		_, _, next_act_prob, next_log_prob = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount

		if self.cfg.action == 'mcts' and self.cfg.reanalyze == True:
			_next_z = next_z.reshape(-1, self.cfg.latent_dim)
			search_root_values, _, _ = self.mcts.search(self.model, _next_z.shape[0], _next_z, task, 0, 
																		 self.cfg.mcts_temperature, self.discount, False, self.device)
			search_root_values = torch.Tensor(search_root_values.reshape(next_z.shape[0], next_z.shape[1], 1)).to(next_z.device)
			td_targets = reward + discount * search_root_values * (1 - done)
			return td_targets
		
		actions = torch.eye(self.cfg.action_dim, device=next_z.device).unsqueeze(0)
		if next_z.dim() == 2:
			# z (batch_size, latent_dim) -> (batch_size, action_dim, latent_dim)
			next_z = next_z.unsqueeze(1).expand(-1, self.cfg.action_dim, -1)
			actions = actions.repeat(next_z.shape[0], 1, 1)
		elif next_z.dim() == 3:
			# z (seq_len, batch_size, latent_dim) -> (seq_len, batch_size, action_dim, latent_dim)
			next_z = next_z.unsqueeze(2).expand(-1, -1, self.cfg.action_dim, -1)
			actions = actions.unsqueeze(0).repeat(next_z.shape[0], next_z.shape[1], 1, 1)
		# encoded_action = encoded_action.squeeze(2)
		qs = self.model.Q(next_z, actions, task, return_type='min', target=True).squeeze(3)

		min_q_next_target = next_act_prob * (qs - self.cfg.entropy_coef * next_log_prob)
		min_q_next_target = min_q_next_target.sum(dim=2, keepdim=True)

		td_targets = reward + discount * min_q_next_target * (1 - done)
		return td_targets

	def _update(self, obs, action, reward, task=None, done = None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task) if not self.cfg.has_done else self._td_target_term(next_z, reward, done, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		pi_loss, pi_grad_norm = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"pi_loss": pi_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"pi_grad_norm": pi_grad_norm,
			"pi_scale": self.scale.value,
		}).detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		done = None
		if self.cfg.has_done:
			obs, action, reward, done, task = buffer.sample()
		else:
			obs, action, reward, task = buffer.sample()
		
		kwargs = {}
		if task is not None:
			kwargs["task"] = task

		if done is not None:
			kwargs["done"] = done
			
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, **kwargs)
	
	def reset_parameters(self):
		self.model.reset(self.cfg)
		self.model.to(self.device)
		self.scale.value= torch.nn.Buffer(torch.ones(1, dtype=torch.float32, device=torch.device(self.cfg.get('device', 'cuda:0'))))
		if self.cfg.optimizer == 'adam':
			self.optim = torch.optim.Adam([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				 }
			], lr=self.cfg.lr, capturable=True)
			self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

		elif self.cfg.optimizer == 'sgd':
			self.optim = torch.optim.SGD([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				 }
			], lr=self.cfg.lr, weight_decay=0.0001,momentum=0.9)
			self.pi_optim = torch.optim.SGD(self.model._pi.parameters(), lr=self.cfg.lr, weight_decay=0.0001,momentum=0.9)

		self.model.eval()
		self.scale = RunningScale(self.cfg)