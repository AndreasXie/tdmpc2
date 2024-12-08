import copy 
import torch
from torch.cuda.amp import autocast as autocast
import numpy as np
from torch import nn
import math
import torch.nn.functional as F
from torch.distributions import TransformedDistribution, Normal, constraints
from torch.distributions import transforms
try:
    from mcts.base import MCTS
    from mcts.math_mcts import *
except:
    from base import MCTS
    from math_mcts import *

class Node:
    discount = 0
    num_actions = 0

    @staticmethod
    def set_static_attributes(discount, num_actions):
        Node.discount = discount
        Node.num_actions = num_actions

    def __init__(self, prior, action=None, parent=None):
        self.prior = prior
        self.action = action
        self.parent = parent

        self.depth = parent.depth + 1 if parent else 0
        self.visit_count = 0
        self.reward = 0.

        self.state = None
        self.estimated_value_lst = []
        self.children = []
        self.selected_children_idx = []

        self.epsilon = 1e-6

        assert Node.num_actions > 1
        assert 0 < Node.discount <= 1.

    def expand(self, state, reward, policy_logits):
        self.state = state
        self.reward = reward

        for action in range(Node.num_actions):
            prior = policy_logits[action]
            child = Node(prior, action, self)
            self.children.append(child)

    def get_policy(self):
        logits = np.asarray([child.prior for child in self.children])
        return softmax(logits)

    def get_normalized_Q(self, normalize_func):
        normalized_Qs = []
        for action, child in enumerate(self.children):
            if child.is_expanded():
                completed_Q = child.get_reward() + self.discount * child.get_value()
            else:
                completed_Q = 0
            # 归一化
            normalized_Qs.append(normalize_func(completed_Q))
        return np.asarray(normalized_Qs)

    def get_children_priors(self):
        return np.asarray([child.prior for child in self.children])

    def get_children_visits(self):
        return np.asarray([child.visit_count for child in self.children])

    def get_children_visit_sum(self):
        visit_lst = self.get_children_visits()
        visit_sum = np.sum(visit_lst)
        assert visit_sum == self.visit_count - 1
        return visit_sum

    def get_value(self):
        if self.is_expanded():
            return np.mean(self.estimated_value_lst)
        else:
            return 0

    def get_qsa(self, action):
        child = self.children[action]
        assert child.is_expanded()
        qsa = child.get_reward() + Node.discount * child.get_value()
        return qsa

    def get_reward(self):
        return self.reward

    def get_root(self):
        node = self
        while not node.is_root():
            node = node.parent
        return node

    def get_expanded_children(self):
        assert self.is_expanded()

        children = []
        for _, child in enumerate(self.children):
            if child.is_expanded():
                children.append(child)
        return children

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        assert self.is_expanded()
        return len(self.get_expanded_children()) == 0

    def is_expanded(self):
        assert (len(self.children) > 0) == (self.visit_count > 0)
        return len(self.children) > 0

    def print(self, info):
        if not self.is_expanded():
            return

        for i in range(self.depth):
            print(info[i], end='')

        is_leaf = self.is_leaf()
        if is_leaf:
            print('└──', end='')
        else:
            print('├──', end='')

        print(self.__str__())

        for child in self.get_expanded_children():
            c = '   ' if is_leaf else '|    '
            info.append(c)
            child.print(info)

    def __str__(self):
        if self.is_root():
            action = self.selected_children_idx
        else:
            action = self.action
        #process numpy to float
        reward = self.get_reward()
        reward = reward.item() if not isinstance(reward,float) else reward
        value = self.get_value()
        value = value.item() if not isinstance(value,float) else value

        s = '[a={} prior = {} (n={}, r={:.3f}, v={:.3f})]'.format(
                action, self.prior, self.visit_count, reward, value)
        return s

class PyMCTS(MCTS):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.c_1 = cfg.c_1
        self.c_2 = cfg.c_2
        self.dirichlet_alpha = cfg.dirichlet_alpha
        self.explore_frac = cfg.explore_frac

    # def expectation(self, values, visits):

    def search(self, model, batch_size, state, task, verbose=0, device="cuda"):
        # 准备工作
        Node.set_static_attributes(self.discount, self.num_actions)  # 设置 MCTS 的静态参数
        roots = [Node(prior=1) for _ in range(batch_size)]          # 为批次设置根节点

        #initial inference
        _, policy, _logits, _ = model.pi(state, task) # action probs
        
        root_states = state.detach().cpu()            # 根节点的状态
        root_policy_logits = _logits.detach().cpu()  # 根节点的策略

        dirichlet_noise = torch.distributions.Dirichlet(torch.tensor([self.dirichlet_alpha]*self.num_actions))

        # 扩展根节点并更新统计信息
        for root, state, logit in zip(roots, root_states, root_policy_logits):
            # noise = np.random.dirichlet([self.dirichlet_alpha] * self.num_actions)
            noise = dirichlet_noise.sample().cpu()
            for action in range(self.num_actions):
                logit[action] = logit[action] * (1 - self.explore_frac) + noise[action] * self.explore_frac

            root.expand(state, np.array([0.]), logit)
            root.visit_count += 1

        value_min_max_lst = [MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]

        self.verbose = verbose

        # 进行 N 次模拟
        mcts_info = {}
        for simulation_idx in range(self.num_simulations):
            leaf_nodes = []                                         # 当前模拟的叶子节点
            last_actions = []                                       # 叶子节点的选择动作
            current_states = []                                     # 叶子节点的隐藏状态
            search_paths = []                                       # 当前搜索迭代中的节点路径

            self.log('Iteration {} \t'.format(simulation_idx), verbose=2, iteration_begin=True)
            if self.verbose > 1:
                self.log('Tree:', verbose=2)
                roots[0].print([])

            # 为根节点选择动作
            trajectories = []
            for idx in range(batch_size):
                node = roots[idx]                                   # 搜索从根节点开始
                search_path = [node]                                # 保存从根到叶子的搜索路径
                value_min_max = value_min_max_lst[idx]              # 记录树状态的最小值和最大值

                # 从根节点搜索直到未扩展的叶子节点
                action = -1
                select_action_lst = []
                while node.is_expanded():
                    action = self.select_action(node, value_min_max)
                    node = node.children[action]
                    search_path.append(node)
                    select_action_lst.append(action)

                assert action >= 0

                self.log('selection path -> {}'.format(select_action_lst), verbose=4)
                # 更新一些统计信息
                parent = search_path[-2]                            # 获取叶子节点的父节点
                current_states.append(parent.state)

                last_actions.append(int_to_one_hot(action,self.num_actions))
                leaf_nodes.append(node)
                search_paths.append(search_path)
                trajectories.append(select_action_lst)

            # 使用模型预测当前状态、奖励、值和策略
            current_states = torch.stack(current_states, dim=0).to(device)
            last_actions = torch.tensor(last_actions,dtype=torch.float32).to(device)

            next_states, next_reward, next_values, next_logits = self.update_statistics(
                prediction=True,                                    # 使用模型预测而不是环境模拟
                model=model,                                        # 模型
                states=current_states,                              # 当前状态
                actions=last_actions,                               # 最后选择的动作
                task=task
            )

            # 扩展叶子节点并向后传播以更新统计信息
            for idx in range(batch_size):
                # 扩展叶子节点
                leaf_nodes[idx].expand(next_states[idx], next_reward[idx], next_logits[idx])
                # 从叶子节点向根节点传播以更新统计信息
                self.back_propagate(search_paths[idx], next_values[idx], value_min_max_lst[idx])

            # 获取最终结果和信息
        search_root_values = np.asarray([root.get_value() for root in roots])
        search_root_actions = []
        for root, value_min_max in zip(roots, value_min_max_lst):
            action = self.final_action(root, temperature=1.0)
            search_root_actions.append(action)
        search_best_actions = np.asarray(search_root_actions)

        if self.verbose:
            self.log('Final Tree:', verbose=1)
            if batch_size == 1:
                roots[0].print([])
            self.log('search root value -> \t\t {} \n'
                     'search best action -> \t\t {}'
                     ''.format(search_root_values[0], search_best_actions[0]),
                     verbose=1, iteration_end=True)
        return search_root_values, search_best_actions, mcts_info


    def final_action(self, root: Node, temperature=1.0):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = root.get_children_visits()
        actions = [child.action for child in root.children]

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)
        return action
    
    def select_action(self, node: Node, value_min_max: MinMaxStats):
        policy = np.expand_dims(node.get_policy(),1)
        Q_score = node.get_normalized_Q(value_min_max.normalize)

        Nb = node.get_children_visit_sum()
        Na = np.expand_dims(node.get_children_visits(),1)

        pb_c = np.sqrt(Nb)/(Na+1)
        pb_c *= self.c_1 + np.log((Nb + self.c_2 + 1)/self.c_2)

        #Q_score act_num, policy act_num, Nb 1, Na act_num,
        children_scores = Q_score + policy * pb_c

        action = np.argmax(children_scores)

        assert action < self.num_actions

        self.log('action select at non-root node: \n'
                     'policy -> \t {} \n'
                     'children q scores -> \t {} \n'
                     'children visits -> \t {} \n'
                     'children scores -> \t {} \n'
                     'best action -> \t {} \n'
                     ''.format(str(policy).replace("\n"," "), str(Q_score).replace("\n"," "), 
                               str(Na).replace("\n"," "), str(children_scores).replace("\n"," "), 
                               str(action).replace("\n"," ")), verbose=4)

        return action

    def back_propagate(self, search_path, leaf_node_value, value_min_max):
        value = leaf_node_value

        for node in reversed(search_path):
            node.estimated_value_lst.append(value)
            node.visit_count += 1

            value_min_max.update(node.get_reward() + node.discount*node.get_value())
            value = node.get_reward() + node.discount* value

class SimpleEnv:
    def __init__(self, state_dim=4, num_actions=3):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.counter = 0

    def reset(self, batch_size=1):
        # 返回固定初始状态，全0张量
        self.current_states = torch.zeros(batch_size, self.state_dim)
        self.counter = 0
        return self.current_states

    def step(self, actions):
        # 每次step让状态加1，确定性
        batch_size = actions.shape[0]
        self.counter += 1
        next_states = torch.ones(batch_size, self.state_dim)*self.counter
        self.current_states = next_states
        return next_states

class SimpleModel(nn.Module):
    def __init__(self, state_dim=4, num_actions=3):
        super(SimpleModel, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

    def forward(self, states):
        # 简单映射，不用于本示例
        return states

    def next(self, states, actions, task=None):
        # next_states = states + 1 的确定性映射
        next_states = states + 1.
        return next_states

    def reward(self, states, actions, task=None):
        # 确定性奖励：r = state[:,0] + action
        # state[:,0] 是batch中state第一个元素
        r = states[:,0] + actions.squeeze(-1).float()
        return r

    def Q(self, states, actions, task=None, return_type='avg'):
        # 确定性Q值：Q = state[:,0]*10 + action
        Q = states[:,0]*10.0 + actions.squeeze(-1).float()
        return Q

    def pi(self, states, task=None):
        # 返回固定策略logits，如： [0.3,0.5,0.2]
        # 不涉及随机函数，固定为常量张量
        logits = torch.tensor([[0.3, 0.5, 0.2]], dtype=torch.float)
        # broadcast到当前batch大小
        batch_size = states.shape[0]
        logits = logits.expand(batch_size, -1)
        return torch.zeros_like(logits), torch.ones_like(logits)*0.1, logits


if __name__ == "__main__":
    class CFG:
        action_dim = 3
        num_simulations = 10
        c_1 = 1.25
        c_2 = 19652.
        discount = 0.99
        value_minmax_delta = 0.01
        value_support = True
        reward_support = True
        value_prefix = True
        horizon = 10
        task_platform = 'SimpleEnv'
        vis = None
        std_magnification = 1.0

    env = SimpleEnv(state_dim=4, num_actions=3)
    model = SimpleModel(state_dim=4, num_actions=3)
    cfg = CFG()
    mcts = PyMCTS(cfg)

    batch_size = 1
    root_states = env.reset(batch_size=batch_size)
    root_values = model.Q(root_states, torch.zeros(batch_size,1).long()).detach().cpu().numpy()
    # 固定策略logits，如上model中定义的pi()
    # pi给定为[0.3,0.5,0.2]，对batch进行扩展
    fixed_logits = np.array([[0.3,0.5,0.2]],dtype=np.float32)

    search_root_values, search_best_actions, mcts_info = \
        mcts.search(model, 
                    batch_size, 
                    root_states, 
                    root_values, 
                    fixed_logits,
                    task=None,
                    verbose=4)

    print("MCTS Search Results:")
    print("Root Values: ", search_root_values)
    print("Best Actions: ", search_best_actions)