from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
from envs import make_env

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			self.env = make_env(self.cfg)
			self.cfg.seed = self.cfg.seed + 5
			done, ep_reward, t = False, 0, 0
			obs = self.env.reset()
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action, _ = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, trunc, info = self.env.step(action)
				done = info['real_done'] if self.cfg.episode_life else done
				ep_reward += info['raw_reward'] if self.cfg.clip_rewards else reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step+i)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None, done=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float(-1))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if done is None:
			done = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			done=done.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next, real_done = {}, True, False, True
		episode_reward = []
		enough_train = False
		prob_entropy = 0
		while self._step <= self.cfg.steps:
			prob_entropy = 0
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True
				self.cfg.seed = self.cfg.seed + 5

			# Reset environment
			if done:
				if real_done:
					self.cfg.max_episode_steps = 27000 if eval_next else 3000
					self.env = make_env(self.cfg)

				if eval_next and real_done:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					enough_train = True
					train_metrics.update(
						episode_reward=torch.tensor(episode_reward).sum(),
						episode_success=info['success'],
						prob_entropy=prob_entropy
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds)) if self._step <= 100_000 else self._ep_idx +1

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]
				episode_reward = []

			# Collect experience
			if self._step <= 100_000:
				if self._step < self.cfg.pretrain_steps:
					action = self.env.rand_act()
				else:
					action, prob_entropy = self.agent.act(obs, t0=len(self._tds)==1)
					action = action.squeeze(0)

				obs, reward, done, trunc, info = self.env.step(action)
				self._tds.append(self.to_td(obs, action, reward, done))
				real_done = info['real_done'] if self.cfg.episode_life else done
				episode_reward.append(info['raw_reward'] if self.cfg.clip_rewards else reward)
			else:
				done = True
				real_done = True

			# Update agent
			if self._step >= self.cfg.pretrain_steps and enough_train:
				if self._step*self.reset_ratio % self.cfg.reset_interval == 0:
					self.agent.reset_parameters()

				for _ in range(self.cfg.replay_ratio):
					_train_metrics = self.agent.update(self.buffer)
					train_metrics.update(_train_metrics)
			self._step += 1

		self.logger.finish(self.agent)