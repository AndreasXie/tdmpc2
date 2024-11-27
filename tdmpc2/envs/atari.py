import gymnasium as gym
import numpy as np

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    FireResetEnv,
    EpisodicLifeEnv as sEpisodicLifeEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

#efficientZero v2 handling of the Atari environment
# class TimeLimit(gym.Wrapper):
#     def __init__(self, env, max_episode_steps=None):
#         super(TimeLimit, self).__init__(env)
#         self._max_episode_steps = max_episode_steps
#         self._elapsed_steps = 0

#     def step(self, ac):
#         observation, reward, done, info = self.env.step(ac)
#         self._elapsed_steps += 1
#         if self._elapsed_steps >= self._max_episode_steps:
#             done = True
#             info['TimeLimit.truncated'] = True
#         return observation, reward, done, info

#     def reset(self, **kwargs):
#         self._elapsed_steps = 0
#         return self.env.reset(**kwargs)


# class NoopResetEnv(gym.Wrapper):
#     def __init__(self, env, noop_max=30):
#         """Sample initial states by taking random number of no-ops on reset.
#         No-op is assumed to be action 0.
#         """
#         gym.Wrapper.__init__(self, env)
#         self.noop_max = noop_max
#         self.override_num_noops = None
#         self.noop_action = 0
#         assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

#     def reset(self, **kwargs):
#         """ Do no-op action for a number of steps in [1, noop_max]."""
#         self.env.reset(**kwargs)
#         if self.override_num_noops is not None:
#             noops = self.override_num_noops
#         else:
#             try:
#                 noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
#             except:
#                 noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
#         assert noops > 0
#         obs = None
#         for _ in range(noops):
#             obs, _, done, _ = self.env.step(self.noop_action)
#             if done:
#                 obs = self.env.reset(**kwargs)
#         return obs

#     def step(self, ac):
#         return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        info['real_done'] = self.was_real_done
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info

    def render(self,mode):
        return self.env.render(mode)


# class MaxAndSkipEnv(gym.Wrapper):
#     def __init__(self, env, skip=4):
#         """Return only every `skip`-th frame"""
#         gym.Wrapper.__init__(self, env)
#         # most recent raw observations (for max pooling across time steps)
#         self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
#         self._skip       = skip
#         self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

#     def step(self, action):
#         """Repeat action, sum reward, and max over last observations."""
#         total_reward = 0.0
#         done = None
#         for i in range(self._skip):
#             obs, reward, done, info = self.env.step(action)
#             if i == self._skip - 2: self._obs_buffer[0] = obs
#             if i == self._skip - 1: self._obs_buffer[1] = obs
#             total_reward += reward
#             if done:
#                 break
#         # Note that the observation on the done=True frame
#         # doesn't matter
#         self.max_frame = self._obs_buffer.max(axis=0)

#         return self.max_frame, total_reward, done, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

#     def render(self, mode='rgb_array', **kwargs):
#         img = self.max_frame
#         img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA).astype(np.uint8)
#         if mode == 'rgb_array':
#             return img
#         elif mode == 'human':
#             from gym.envs.classic_control import rendering
#             if self.viewer is None:
#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(img)
#             return self.viewer.isopen


# class WarpFrame(gym.ObservationWrapper):
#     def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
#         """
#         Warp frames to 84x84 as done in the Nature paper and later work.
#         If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
#         observation should be warped.
#         """
#         super().__init__(env)
#         self._width = width
#         self._height = height
#         self._grayscale = grayscale
#         self._key = dict_space_key
#         if self._grayscale:
#             num_colors = 1
#         else:
#             num_colors = 3

#         new_space = gym.spaces.Box(
#             low=0,
#             high=255,
#             shape=(self._height, self._width, num_colors),
#             dtype=np.uint8,
#         )
#         if self._key is None:
#             original_space = self.observation_space
#             self.observation_space = new_space
#         else:
#             original_space = self.observation_space.spaces[self._key]
#             self.observation_space.spaces[self._key] = new_space
#         assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

#     def observation(self, obs):
#         if self._key is None:
#             frame = obs
#         else:
#             frame = obs[self._key]

#         if self._grayscale:
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(
#             frame, (self._width, self._height), interpolation=cv2.INTER_AREA
#         )
#         if self._grayscale:
#             frame = np.expand_dims(frame, -1)

#         if self._key is None:
#             obs = frame
#         else:
#             obs = obs.copy()
#             obs[self._key] = frame
#         return obs
    
#     #weird bug, observation doesnt work automatically, gym 0.26.0
#     def reset(self, **kwargs):
#         return self.observation(self.env.reset(**kwargs))

class SimpleWrapper(gym.Wrapper):
    def __init__(self, cfg, env):
        """Cosine Consistency loss function: similarity loss
        Parameters
        ----------
        obs_to_string: bool. Convert the observation to jpeg string if True, in order to save memory usage.
        """
        super().__init__(env)
        self.clip_reward = cfg.get('clip_reward')
        self.action_range = env.action_space.n
        self.thresholds = np.linspace(0, 1, self.action_range+1)
    
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(np.argmax(action).item())

        info['raw_reward'] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, 1. if done or trunc else 0., info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def close(self):
        return self.env.close()
    
    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode, **kwargs)

def make_atari(cfg):
    """Make Atari games
    Parameters
    ----------
    game_name: str
        name of game (Such as Breakout, Pong)
    kwargs: dict
        skip: int
            frame skip
        obs_shape: (int, int)
            observation shape
        gray_scale: bool
            use gray observation or rgb observation
        seed: int
            seed of env
        max_episode_steps: int
            max moves for an episode
        save_path: str
            the path of saved videos; do not save video if None
            :param seed:
        game_name, seed, save_path=None, **kwargs
    """
    # params
    env_id = cfg.get('task')
    gray_scale = cfg.get('gray_scale')
    obs_to_string = cfg.get('obs_to_string')
    skip = cfg.get('n_skip', 4)
    obs_shape = cfg.get('obs_shape') if cfg.get('obs_shape') != '???' else [3, 96, 96]
    resize = cfg.get('resize', 96)
    max_episode_steps = cfg.get('max_episode_steps') if cfg.get('max_episode_steps') else 108000 // skip
    episodic_life = cfg.get('episode_life')
    clip_reward = cfg.get('clip_rewards')
    noop = cfg.get('noop')

    try:
        if "v1" in env_id:#which means the game is from the classic tasks of gym like cartpole and  no pixel wrapper is needed
            env = gym.make(env_id)
            env = SimpleWrapper(cfg,env)
        else:

            env = gym.make(env_id) 
            env = gym.wrappers.RecordEpisodeStatistics(env)

            env = NoopResetEnv(env, noop_max=noop) if noop > 0 else env
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)#manually made, rather than using 
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env) if gray_scale else env
            env = gym.wrappers.FrameStack(env, 4)
            env.action_space.seed(cfg.get('seed'))
        return env
    except:
        raise ValueError('Unknown task:', env_id)
    
#Just for tesing
def run_one_episode(env_cfg):
    """
    Run one episode in the Atari environment with random actions and print step-by-step information.
    
    Parameters:
        env_cfg (dict): Configuration dictionary for the environment.
    """
    # Create the Atari environment
    env = make_atari(env_cfg)
    print(f"Environment created: {env_cfg.get('task')}")

    # Reset the environment
    print("\nStarting episode...")
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape if isinstance(observation, np.ndarray) else type(observation)}")

    total_reward = 0
    step = 0

    while True:
        # Sample a random action
        action = env.action_space.sample()

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Print step information
        print(f"Step {step}:")
        print(f"  Action Taken: {action}")
        print(f"  Reward Received: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Info: {info}")
        print(f"  Observation Shape: {observation.shape if isinstance(observation, np.ndarray) else type(observation)}")
        
        a = env.render('rgb_array')
        # End the loop if the episode is terminated or truncated
        if info['lives']==0:
            info['episode']
            print("\nEpisode finished.")
            print(f"Total Steps: {step}")
            print(f"Total Reward: {total_reward}")
            break

    # Close the environment
    env.close()
    print("Environment closed successfully.")

if __name__ == "__main__":
    # Example configuration for testing Breakout
    cfg = {
        'task': 'AsterixNoFrameskip-v4',
        'gray_scale': True,
        'obs_to_string': False,
        'n_skip': 4,
        'obs_shape': [3, 96, 96],
        'resize': 84,
        'max_episode_steps': 1000,
        'episode_life': True,
        'clip_rewards': True,
        'seed': 42,
        'noop':0
    }

    run_one_episode(cfg)