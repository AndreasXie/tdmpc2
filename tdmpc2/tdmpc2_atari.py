import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from common import math, init, layers
from common.scale import RunningScale, RunningMeanStd
from common.world_model_atari import WorldModel
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
				{'params': self.model._termination.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				 }
			], lr=self.cfg.lr,weight_decay=0.01 ,capturable=True)
			self.pi_optim = torch.optim.AdamW(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

		elif cfg.optimizer == 'sgd':
			self.optim = torch.optim.SGD([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._termination.parameters()},
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
		return self.cfg.get('discount', 0.997)

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
		kl_div = 0
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			action, prob_entropy,kl_div = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
			prob_entropy = prob_entropy.mean()
		else:
			z = self.model.encode(obs, task)
			action = self.model.pi(z, task)[1]
			if self.cfg.get('task_platform') == 'atari':
				action = action.squeeze(0) # TODO: this is a bit hacky
		return action.cpu(), prob_entropy, kl_div

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""
		Estimate value of a trajectory starting at latent state z and executing given actions.
		terminated = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G += discount * (1-terminated) * reward
			discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			terminated = torch.clip_(terminated + (self.model.terminated(z, task) > 0.5).float(), max=1.)
		return G + discount * (1-terminated) * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')
		"""
		G, discount = 0, 1
		terminated = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-terminated) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			terminated = torch.clip_(terminated + (self.model.terminated(z, task) > 0.5).float(), max=1.)
		pi = self.model.pi(z, task)[1]
		if self.cfg.action == 'discrete':
			pi = pi.squeeze(1) # TODO: this is a bit hacky
		return G + discount * (1-terminated) * self.model.Q(z, pi, task, return_type='avg')

	@torch.no_grad()
	def _multistep_randomshooting_gt(self, z, t0=False, eval_mode=False, task=None):
		import itertools
		"""
		Plan a sequence of actions using multi-step MPPI for a discrete action space.
		"""

		action_space = list(range(self.cfg.action_dim))
		all_possible_trajs = list(itertools.product(action_space, repeat=self.cfg.horizon))  # list of tuples, length: A^H
		num_possible = len(all_possible_trajs)
		actions = math.int_to_one_hot(torch.tensor(all_possible_trajs, device=self.device).long(),num_classes=self.cfg.action_dim)  # (A^H, H)
		actions = actions.permute(1,0,2)  # (H, A^H, L)
		z = z[0,:].repeat(num_possible, 1)  # (A^H, L)
		# (c) Optional: apply task mask
		if self.cfg.multitask:
			actions = actions * self.model._action_masks[task]

		# (d) Evaluate Compute advantage value of each trajectory
		values = self._estimate_value(z, actions, task).nan_to_num(0)
		values = values - values.mean()
		value = torch.exp(values - values.max())
		value = (value*actions)
		
		prob = (value.sum(dim=1) / (value.sum()/self.cfg.horizon+1e-6))

		return prob
	
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
				pi_actions[t] = action
				_z = self.model.next(_z, pi_actions[t], task)
			action = self.model.pi(_z, task)[1]
			pi_actions[-1] = action

		# 3) Repeat latent state for all samples
		z = z.repeat(self.cfg.num_samples, 1)

		# 4) Initialize probability distribution over actions (T,A)
		prob_action = torch.full((self.cfg.horizon, self.cfg.action_dim),
								1.0 / self.cfg.action_dim, device=self.device)

		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
					# (a) Fill in policy trajectories if any
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# 5) Multi-step MPPI / random-shooting iterations
		for iteration in range(self.cfg.iterations):
			# (b) Sample the rest from current prob_action
			norm_prob = torch.softmax(prob_action - prob_action.max(dim=1,keepdim=True).values + 1e-6, dim=-1)
			dist = Categorical(probs=norm_prob)  
			actions_sample = dist.sample((self.cfg.num_samples - self.cfg.num_pi_trajs,))  # shape: (N-num_pi, T)
			actions_sample = actions_sample.t()  # shape: (T, N-num_pi)
			actions[:, self.cfg.num_pi_trajs:] = math.int_to_one_hot(actions_sample, self.cfg.action_dim)

			# (c) Optional: apply task mask
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# (d) Evaluate Compute advantage value of each trajectory
			values = self._estimate_value(z, actions, task).nan_to_num(0)
			# values = values - values.mean()
			value = torch.exp(values - values.max())

			if self.cfg.gradient:
				importance_dist = (actions.permute(1,0,2) - norm_prob).permute(1,0,2)

				weighted_value = value * importance_dist

				prob_gradient = weighted_value.sum(dim=1) / (value.sum() / self.cfg.horizon+1e-6)

				# (h) Blend with old distribution
				prob_action = prob_action + self.cfg.alpha  * prob_gradient 
			else:
				# (e) Compute new distribution
				new_prob = (value * actions).sum(dim=1) / (value.sum() / self.cfg.horizon+1e-6)

				prob_action = (1 - self.cfg.alpha) * prob_action + self.cfg.alpha * new_prob

			# (i) Add Dirichlet noise for exploration
			explora_frac = self.cfg.explore_frac/(iteration+1)
			dirichlet_noise = torch.distributions.Dirichlet(torch.tensor([self.cfg.dirichlet_alpha] 
																* prob_action.shape[-1]
																* self.cfg.horizon )).sample().to(self.device).reshape(self.cfg.horizon, self.cfg.action_dim)
			prob_action= (1 - explora_frac) * prob_action+ explora_frac * dirichlet_noise

		if eval_mode and self.cfg.eval_perf:
			gt_prob = self._multistep_randomshooting_gt(z, t0=t0, eval_mode=eval_mode, task=task)
			norm_prob = norm_prob = torch.softmax(prob_action - prob_action.max(dim=1,keepdim=True).values + 1e-6, dim=-1)
			norm_gt_prob = torch.softmax(gt_prob - gt_prob.max(dim=1,keepdim=True).values + 1e-6, dim=-1)
			kl_div = torch.nn.functional.kl_div(norm_prob.log(), norm_gt_prob, reduction='sum')
		else:
			kl_div = torch.tensor(0.0)

		# 6) After finishing all iterations, pick or sample the first action
		norm_prob = torch.softmax(prob_action - prob_action.max(dim=1,keepdim=True).values + 1e-6, dim=-1)
		dist = Categorical(probs=norm_prob) 
		if self.cfg.rs_argmax: 
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
			score = score / (score.sum(0)+ 1e-6)

			tau = 0.5 if eval_mode else 2.0
			rand_idx = math.gumbel_softmax_sample(score.squeeze(1),temperature=tau)  # gumbel_softmax_sample is compatible with cuda graphs
			actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		else:
			#Directly sample from the distribution
			actions = dist.sample().squeeze(0)
			actions = math.int_to_one_hot(actions, self.cfg.action_dim)

		entropy = dist.entropy()

		return actions[0], entropy, kl_div if eval_mode else None

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
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Random shooting
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

		return pi_loss.detach(), pi_grad_norm
	
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
			td_targets = self._td_target_term(next_z, reward, done, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.n_step+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)

		zs[0] = z
		consistency_loss = 0
		continuity_loss = 0
		for t, (_action, _next_h) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss - torch.cosine_similarity(z, _next_h).mean() * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		terminated_pred = self.model.terminated(zs[-1], task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t
		terminated_loss = F.binary_cross_entropy(terminated_pred, done)
		consistency_loss = consistency_loss / self.cfg.n_step
		reward_loss = reward_loss / self.cfg.n_step
		value_loss = value_loss / (self.cfg.n_step * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.terminated_coef * terminated_loss +
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
			"terminated_loss": terminated_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"pi_grad_norm": pi_grad_norm,
			"pi_scale": self.scale.value,
		}).detach().mean()
	
	@torch.no_grad()
	def _td_target_term_n_step(self, next_z, rewards, dones, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""

		n_step = rewards.shape[0]
		gamma = self.discount.unsqueeze(0) if self.cfg.multitask else self.discount

		# 计算折扣累积奖励
		discounted_rewards = torch.zeros_like(rewards[0])
		terminated = torch.zeros(dones.shape[1], 1, dtype=torch.float32, device=dones.device)
		terminated = dones[0]
		for k in range(n_step):
			discounted_rewards += (gamma ** k) * rewards[k]
			discounted_rewards = discounted_rewards * (1 - terminated)
			terminated = torch.clip_(terminated + dones[k], max=1.)

		# 获取n步后的潜在状态
		final_next_z = next_z[-1]  # 形状: (batch_size, latent_dim)

		# 计算n步后的Q值
		_, _, next_act_prob, next_log_prob = self.model.pi(final_next_z, task)

		actions = torch.eye(self.cfg.action_dim, device=next_z.device).unsqueeze(0)
		if final_next_z.dim() == 2:
		    # (batch_size, latent_dim) -> (batch_size, action_dim, latent_dim)
			final_next_z = final_next_z.unsqueeze(1).expand(-1, self.cfg.action_dim, -1)
			actions = actions.repeat(final_next_z.shape[0], 1, 1)
		elif final_next_z.dim() == 3:
		    # (seq_len, batch_size, latent_dim) -> (seq_len, batch_size, action_dim, latent_dim)
			final_next_z = final_next_z.unsqueeze(2).expand(-1, -1, self.cfg.action_dim, -1)
			actions = actions.unsqueeze(0).repeat(final_next_z.shape[0], final_next_z.shape[1], 1, 1)

		qs = self.model.Q(final_next_z, actions, task, return_type='min', target=True).squeeze(2)

		min_q_next_target = next_act_prob * (qs - self.cfg.entropy_coef * next_log_prob)
		min_q_next_target = min_q_next_target.sum(dim=1, keepdim=True)

		# 计算n步TD目标
		td_targets = discounted_rewards + (gamma ** n_step) * min_q_next_target * (1 - dones[-1])

		return td_targets


	def _update_n_step(self, obs, action, reward, task=None, done = None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target_term_n_step(next_z, reward, done, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.n_step+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)

		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss - torch.cosine_similarity(z, _next_z).mean() * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs[0], action[0], task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		terminated_pred = self.model.terminated(_zs, task)
		
		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t

		for i in range(self.cfg.num_q):
			value_loss = value_loss + math.soft_ce(qs[i], td_targets, self.cfg).mean()

		terminated_loss = F.binary_cross_entropy(terminated_pred, done)
		consistency_loss = consistency_loss / self.cfg.n_step
		reward_loss = reward_loss / self.cfg.n_step
		value_loss = value_loss / self.cfg.num_q

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.terminated_coef * terminated_loss +
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
			"terminated_loss": terminated_loss,
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
		obs, action, reward, done, task = buffer.sample()

		kwargs = {}
		if task is not None:
			kwargs["task"] = task

		if done is not None:
			kwargs["done"] = done
			
		torch.compiler.cudagraph_mark_step_begin()
		if self.cfg.n_step_return == True:
			stats = self._update_n_step(obs, action, reward, **kwargs)
		else:
			stats = self._update(obs, action, reward, **kwargs)

		return stats
	
	def reset_parameters(self):
		self.model.reset(self.cfg)
		self.model.to(self.device)
		self.scale.value= torch.nn.Buffer(torch.ones(1, dtype=torch.float32, device=torch.device(self.cfg.get('device', 'cuda:0'))))
		if self.cfg.optimizer == 'adam':
			self.optim = torch.optim.AdamW([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._termination.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				 }
			], lr=self.cfg.lr, capturable=True)
			self.pi_optim = torch.optim.AdamW(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

		elif self.cfg.optimizer == 'sgd':
			self.optim = torch.optim.SGD([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._termination.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				 }
			], lr=self.cfg.lr, weight_decay=0.0001,momentum=0.9)
			self.pi_optim = torch.optim.SGD(self.model._pi.parameters(), lr=self.cfg.lr, weight_decay=0.0001,momentum=0.9)

		self.model.eval()
		self.scale = RunningScale(self.cfg)