import gymnasium as gym
import numpy as np

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    FireResetEnv,
    EpisodicLifeEnv as sEpisodicLifeEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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
        
        # a = env.render('rgb_array')
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