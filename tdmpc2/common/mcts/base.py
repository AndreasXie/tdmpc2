import copy
import torch
from torch.cuda.amp import autocast as autocast
import numpy as np
from ..math import two_hot_inv

class MCTS:
    def __init__(self, cfg):
        self.num_actions = cfg.num_actions
        self.num_simulations = cfg.num_simulations
        self.num_top_actions = cfg.num_top_actions
        self.c_visit = cfg.c_visit
        self.c_scale = cfg.c_scale
        self.c_base = cfg.c_base
        self.c_init = cfg.c_init
        self.dirichlet_alpha = cfg.dirichlet_alpha
        self.explore_frac = cfg.explore_frac
        self.discount = cfg.discount
        self.value_minmax_delta = cfg.value_minmax_delta
        self.value_support = cfg.value_support
        self.reward_support = cfg.reward_support
        self.value_prefix = cfg.value_prefix
        self.hidden_horizon_len = 1
        self.mpc_horizon = cfg.mpc_horizon
        self.env = cfg.env
        self.vis = cfg.vis  # vis: [log, text, graph]
        self.std_magnification = cfg.std_magnification

        self.current_num_top_actions = self.num_top_actions  # /2 every phase
        self.current_phase = 0  # current phase index
        self.visit_num_for_next_phase = max(
            np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.current_num_top_actions)), 1
        ) * self.current_num_top_actions  # how many visit counts for next phase
        self.used_visit_num = 0
        self.verbose = 0
        assert self.num_top_actions <= self.num_actions

    def search(self, model, batch_size, root_states, root_values, root_policy_logits, **kwargs):
        raise NotImplementedError()

    def sample_mpc_actions(self, policy):
        is_continuous = (self.env in ['DMC', 'Gym'])
        if is_continuous:
            action_dim = policy.shape[-1] // 2
            mean = policy[:, :action_dim]
            return mean
        else:
            return policy.argmax(dim=-1).unsqueeze(1)

    def update_statistics(self, **kwargs):
        if kwargs.get('prediction'):
            # prediction for next states, rewards, values, logits
            model = kwargs.get('model')
            states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            task = kwargs.get('task')

            next_value_prefixes = 0
            for _ in range(self.mpc_horizon):
                with torch.no_grad():
                    with autocast():
                        states = model.next(states, last_actions, task)
                        pred_value_prefixes = two_hot_inv(model.reward(states, last_actions, task), self.cfg)
                        next_values = model.Q(states, last_actions, task, return_type='avg')
                        next_logits = model.pi(states,task)[2]
                # last_actions = self.sample_mpc_actions(next_logits)
                next_value_prefixes += pred_value_prefixes

            # process outputs
            next_value_prefixes = next_value_prefixes.detach().cpu().numpy()
            next_values = next_values.detach().cpu().numpy()

            self.log(
                'simulate action {}, r = {:.3f}, v = {:.3f}, logits = {}'.format(
                    last_actions[0].tolist(),
                    next_value_prefixes[0].item(),
                    next_values[0].item(),
                    next_logits[0].tolist()
                ),
                verbose=3
            )
            return states, next_value_prefixes, next_values, next_logits
        else:
            # env simulation for next states
            env = kwargs.get('env')
            current_states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            states = env.step(last_actions)
            raise NotImplementedError()

    def estimate_value(self, **kwargs):
        # prediction for value in planning
        model = kwargs.get('model')
        current_states = kwargs.get('states')
        actions = kwargs.get('actions')

        Value = 0
        discount = 1
        for i in range(actions.shape[0]):
            current_states_hidden = None
            with torch.no_grad():
                with autocast():
                    next_states, pred_value_prefixes, next_values, next_logits = \
                        model.recurrent_inference(current_states, actions[i])

            next_value_prefixes = next_value_prefixes.detach()
            next_values = next_values.detach()
            current_states = next_states
            Value += next_value_prefixes * discount
            discount *= self.discount

        Value += discount * next_values

        return Value

    def log(self, string, verbose, iteration_begin=False, iteration_end=False):
        if verbose <= self.verbose:
            if iteration_begin:
                print('>' * 50)
            print(string)
            print('-' * 20)
            if iteration_end:
                print('<' * 50)

    def reset(self):
        self.current_num_top_actions = self.num_top_actions
        self.current_phase = 0
        self.visit_num_for_next_phase = max(
            np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.current_num_top_actions)), 1
        ) * self.current_num_top_actions  # how many visit counts for next phase
        self.used_visit_num = 0
        self.verbose = 0