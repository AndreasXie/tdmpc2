import gym

def make_simple(cfg):
    """Only for Pendulum-v1, a simple task for debugging.
    """
    # params
    env_id = cfg.get('task')
    env = gym.make(env_id)
    env.seed(1)

    return env