import gym
import numpy as np
import cv2
from collections import deque

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        trunc = False
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            trunc = True
        return observation, reward, done, trunc, info

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
            obs, _, done, info = self.env.step(self.noop_action)
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
        info['real_done'] = self.was_real_done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
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
            obs, reward, done, info = self.env.step(0)
            if done:
                obs = self.env.reset(**kwargs)
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

    def render(self, mode='human', **kwargs):
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
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

class FrameStack(gym.ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally

    """

    def __init__(self, env, num_stack):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.array(list(self.frames))

    def step(self, action):
        observation, reward, done, trunc, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, trunc, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()

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
    env_id = cfg.get('task') + 'NoFrameskip-v4'
    gray_scale = cfg.get('gray_scale')
    skip = cfg.get('n_skip', 4)
    resize = cfg.get('resize', 96)
    max_episode_steps = cfg.get('max_episode_steps')
    episodic_life = cfg.get('episode_life')
    noop = cfg.get('noop')

    try:
        env = gym.make(env_id) 
    except:
        raise ValueError(f"Unknown task: {env_id}")
    env.seed(cfg.get('seed'))
    env = NoopResetEnv(env, noop_max=noop) if noop > 0 else env
    env = MaxAndSkipEnv(env, skip=skip) if skip > 1 else env
    env = EpisodicLifeEnv(env) if episodic_life else env
    env = WarpFrame(env, width=resize, height=resize, grayscale=gray_scale) 
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
    # env = FrameStack(env, skip)
    env.action_space.seed(cfg.get('seed'))
    return env

    
def test_make_atari():
    # 配置参数
    cfg = {
        'task': 'Pong',               # 选择一个 Atari 游戏，如 Pong
        'gray_scale': True,           # 是否使用灰度图像
        'n_skip': 4,                  # 帧跳跃次数
        'resize': 96,                 # 图像缩放尺寸
        'max_episode_steps': 3000,    # 每个 episode 的最大步数
        'episode_life': True,         # 是否使用生命值作为 episode 的结束条件
        'noop': 30,                   # 重置时的无动作步数
        'seed': 42                    # 随机种子
    }

    try:
        # 创建 Atari 环境
        env = make_atari(cfg)
        print("环境创建成功！")
    except ValueError as e:
        print(f"环境创建失败: {e}")
        return

    # 重置环境并获取初始观察值
    obs = env.reset()
    print(f"初始观察值的形状: {obs.shape}")

    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # 随机选择一个动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, done, trunc, info = env.step(action)
        
        # 累积奖励
        total_reward += reward
        step_count += 1
        
        # 每100步打印一次状态
        if step_count % 100 == 0:
            print(f"步数: {step_count}, 累计奖励: {total_reward}")
        
        # 可选：渲染环境（如果需要视觉反馈）
        # env.render()

    print(f"一个 episode 结束！总步数: {step_count}, 总奖励: {total_reward}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    test_make_atari()