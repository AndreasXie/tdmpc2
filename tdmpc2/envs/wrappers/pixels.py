from collections import deque

import gym
import numpy as np
import torch
import common.math as math

class PixelWrapper(gym.Wrapper):
	"""
	Wrapper for pixel observations. Compatible with DMControl environments.
	"""

	def __init__(self, cfg, env, num_frames=3, render_size=64):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, render_size, render_size), dtype=np.uint8
		)
		self._frames = deque([], maxlen=num_frames)
		self._render_size = render_size

	def _get_obs(self):
		frame = self.env.render(
			mode='rgb_array', width=self._render_size, height=self._render_size
		).transpose(2, 0, 1)
		self._frames.append(frame)
		return torch.from_numpy(np.concatenate(self._frames))

	def reset(self):
		self.env.reset()
		for _ in range(self._frames.maxlen):
			obs = self._get_obs()
		return obs

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		return self._get_obs(), reward, done, info


class PixelWrapperAtari(gym.Wrapper):
    def __init__(self, cfg, env):
        """Cosine Consistency loss function: similarity loss
        Parameters
        ----------
        obs_to_string: bool. Convert the observation to jpeg string if True, in order to save memory usage.
        """
        super().__init__(env)
        self.clip_reward = cfg.get('clip_rewards')
        self.action_range = env.action_space.n
        self.transpose = not cfg.gray_scale#if no gray sacle, dim 4,84,84,3 need to be transpose
        self.resize = cfg.get('resize')
        self.gray_scale = cfg.get('gray_scale')
        self.faster_buffer = cfg.get('faster_buffer')

    def format_obs(self, obs):
        if self.transpose:
            if self.faster_buffer:
                obs = obs.permute(2, 0, 1)
            else:
                obs = obs.permute(0, 3, 1, 2).reshape(12, self.resize, self.resize)
        else:
            if self.faster_buffer:
                 obs = obs.reshape(1, self.resize, self.resize)
            else:
                obs = obs.reshape(4, self.resize, self.resize)

        return obs

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(torch.argmax(action))

        obs = self.format_obs(obs)

        info['raw_reward'] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, done, trunc, info

    def rand_act(self):
        action = torch.tensor(self.action_space.sample(), dtype=torch.int64)
        return math.int_to_one_hot(action, self.action_space.n)
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # format observation
        obs = self.format_obs(obs)

        return obs

    def close(self):
        return self.env.close()
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode)
    
    def save_video(self):
        return self.env.save_video()
