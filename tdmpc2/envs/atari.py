import cv2
import gym
import numpy as np
from gym.wrappers import Monitor

#efficientZero v2 handling of the Atari environment
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            try:
                noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
            except:
                noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        info['real_done'] = self.was_real_done
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='rgb_array', **kwargs):
        img = self.max_frame
        img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame

        return obs


def arr_to_str(arr):
    """
    To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings.
    :param arr:
    :return:
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()

    return img_str

class PixelWrapper(gym.Wrapper):
    def __init__(self, env, obs_to_string, clip_reward, action_mode='category'):
        """Cosine Consistency loss function: similarity loss
        Parameters
        ----------
        obs_to_string: bool. Convert the observation to jpeg string if True, in order to save memory usage.
        """
        super().__init__(env)
        self.obs_to_string = obs_to_string
        self.clip_reward = clip_reward
        self.action_range = env.action_space.n
        self.action_mode = action_mode
        self.thresholds = np.linspace(0, 1, self.action_range+1)

    def format_obs(self, obs):
        obs = obs.transpose(2, 0, 1)
        if self.obs_to_string:
            # convert obs to jpeg string for lower memory usage
            obs = obs.astype(np.uint8)
            obs = arr_to_str(obs)
        return obs

    def step(self, action):
        # action = np.clip(np.round(action*self.action_range), 0, self.action_range - 1)
        if self.action_mode == 'category':
            action = np.clip(np.argmax(action),0, self.action_range - 1)
        elif self.action_mode == 'continuous':
            action = np.digitize(action, self.thresholds) - 1
            action = np.clip(action, 0, self.action_range - 1)
        else:#discrete
            pass
        obs, reward, done, info = self.env.step(int(action))

        obs = self.format_obs(obs)

        info['raw_reward'] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # format observation
        obs = self.format_obs(obs)

        return obs

    def close(self):
        return self.env.close()
    
    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode, **kwargs)
class SimpleWrapper(gym.Wrapper):
    def __init__(self, env, clip_reward, action_mode='category'):
        """Cosine Consistency loss function: similarity loss
        Parameters
        ----------
        obs_to_string: bool. Convert the observation to jpeg string if True, in order to save memory usage.
        """
        super().__init__(env)
        self.clip_reward = clip_reward
        self.action_range = env.action_space.n
        self.action_mode = action_mode
        self.thresholds = np.linspace(0, 1, self.action_range+1)
    
    def step(self, action):
        # action = np.clip(np.round(action*self.action_range), 0, self.action_range - 1)
        if self.action_mode == 'category':
            action = np.clip(np.argmax(action),0, self.action_range - 1)
        else:
            action = np.digitize(action, self.thresholds) - 1
            action = np.clip(action, 0, self.action_range - 1)

        obs, reward, done, info = self.env.step(int(action))

        info['raw_reward'] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, done, info

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
    max_episode_steps = cfg.get('max_episode_steps') if cfg.get('max_episode_steps') else 108000 // skip
    episodic_life = cfg.get('episode_life')
    clip_reward = cfg.get('clip_rewards')

    if "v1" in env_id:#which means the game is from the old version of gym and maybe no pixel wrapper is needed
        env = gym.make(env_id)

        env = SimpleWrapper(env, clip_reward=clip_reward, action_mode=cfg.get('action_mode'))
    else:
        env = gym.make(env_id + 'NoFrameskip-v4' if skip == 1 else env_id + 'Deterministic-v4') 

        # random restart
        env = NoopResetEnv(env, noop_max=30)

        # frame skip
        env = MaxAndSkipEnv(env, skip=skip) 

        # episodic trajectory
        if episodic_life:
            env = EpisodicLifeEnv(env)

        # set seed
        env.seed(cfg.get('seed'))
            # reshape size and gray scale
        env = WarpFrame(env, width=obs_shape[1], height=obs_shape[2], grayscale=gray_scale)

            # set max limit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # save video to given
        # if cfg.get('save_video'):
        #     env = Monitor(env, directory="./video", force=True)

        # your wrapper
        env = PixelWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward, action_mode=cfg.get('action_mode'))
    return env