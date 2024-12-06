import copy 
import torch
from torch.cuda.amp import autocast as autocast
import numpy as np
from torch import nn
import math
import torch.nn.functional as F
from torch.distributions import TransformedDistribution, Normal, constraints
from torch.distributions import transforms
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
        self.value_prefix = 0.

        self.state = None
        self.estimated_value_lst = []
        self.children = []
        self.selected_children_idx = []
        self.reset_value_prefix = True
        self.Q_value = []

        self.epsilon = 1e-6

        assert Node.num_actions > 1
        assert 0 < Node.discount <= 1.

    def expand(self, state, value_prefix, policy_logits, reset_value_prefix=True):
        self.state = state
        self.value_prefix = value_prefix
        self.reset_value_prefix = reset_value_prefix

        for action in range(Node.num_actions):
            prior = policy_logits[action]
            child = Node(prior, action, self)
            self.children.append(child)
            self.Q_value.append(0)

    def get_policy(self):
        logits = np.asarray([child.prior for child in self.children])
        return softmax(logits)

    def get_v_mix(self):
        """
        v_mix implementation, refer to https://openreview.net/pdf?id=bERaNdoegnO (Appendix D)
        """
        pi_lst = self.get_policy()
        pi_sum = 0
        pi_qsa_sum = 0

        for action, child in enumerate(self.children):
            if child.is_expanded():
                pi_sum += pi_lst[action]
                pi_qsa_sum += pi_lst[action] * self.get_qsa(action)

        # 如果没有子节点被访问
        if pi_sum < self.epsilon:
            v_mix = self.get_value()
        else:
            visit_sum = self.get_children_visit_sum()
            v_mix = (1. / (1. + visit_sum)) * (self.get_value() + visit_sum * pi_qsa_sum / pi_sum)

        return v_mix

    def get_normalized_Q(self, normalize_func):
        normalized_Qs = []
        for action, child in enumerate(self.children):
            if child.is_expanded():
                completed_Q = self.Q_value[action]
            else:
                completed_Q = 0
            # 归一化
            normalized_Qs.append(normalize_func(completed_Q))
        return np.asarray(normalized_Qs)

    def get_completed_Q(self, normalize_func):
        completed_Qs = []
        v_mix = self.get_v_mix()
        for action, child in enumerate(self.children):
            if child.is_expanded():
                completed_Q = self.get_qsa(action)
            else:
                completed_Q = v_mix
            # 归一化
            completed_Qs.append(normalize_func(completed_Q))
        return np.asarray(completed_Qs)

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
            return self.parent.get_v_mix()

    def get_qsa(self, action):
        child = self.children[action]
        assert child.is_expanded()
        qsa = child.get_reward() + Node.discount * child.get_value()
        return qsa

    def get_reward(self):
        if self.reset_value_prefix:
            return self.value_prefix
        else:
            assert self.parent is not None
            return self.value_prefix - self.parent.value_prefix

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
        value_prefix = self.value_prefix.item() if not isinstance(self.value_prefix,float)else self.value_prefix
        reward = self.get_reward()
        reward = reward.item() if not isinstance(reward,float) else reward
        value = self.get_value()
        value = value.item() if not isinstance(value,float) else value

        s = '[a={} reset={} (n={}, vp={:.3f} r={:.3f}, v={:.3f})]'.format(
            action, self.reset_value_prefix, self.visit_count, value_prefix, reward, value)
        return s

device = "cpu"
# 继承自 MCTS 基类的 PyMCTS 类
class PyMCTS(MCTS):
    def __init__(self, cfg):
        super().__init__(cfg)

    def search(self, model, batch_size, root_states, root_values, root_policy_logits, task, 
               temperature=1.0, verbose=0, **kwargs):
        # 准备工作
        Node.set_static_attributes(self.discount, self.num_actions)  # 设置 MCTS 的静态参数
        roots = [Node(prior=1) for _ in range(batch_size)]          # 为批次设置根节点
        # 扩展根节点并更新统计信息
        for root, state, value, logit in zip(roots, root_states, root_values, root_policy_logits):
            root.expand(state, np.array([0.]), logit)
            root.estimated_value_lst.append(value)
            root.visit_count += 1
        # 保存树节点的最小和最大值
        value_min_max_lst = [MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]

        assert batch_size == len(root_states) == len(root_values)
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
                    action = self.select_action(node, value_min_max, simulation_idx)
                    node = node.children[action]
                    search_path.append(node)
                    select_action_lst.append(action)

                assert action >= 0

                self.log('selection path -> {}'.format(select_action_lst), verbose=4)
                # 更新一些统计信息
                parent = search_path[-2]                            # 获取叶子节点的父节点
                current_states.append(parent.state)

                last_actions.append(action)
                leaf_nodes.append(node)
                search_paths.append(search_path)
                trajectories.append(select_action_lst)

            # 使用模型预测当前状态、奖励、值和策略
            current_states = torch.stack(current_states, dim=0).to(device)
            last_actions = torch.tensor(last_actions).to(device).long().unsqueeze(1)
            next_states, next_value_prefixes, next_values, next_logits = self.update_statistics(
                prediction=True,                                    # 使用模型预测而不是环境模拟
                model=model,                                        # 模型
                states=current_states,                              # 当前状态
                actions=last_actions,                               # 最后选择的动作
                task=task
            )

            # 将 value_prefix 转换为奖励
            search_lens = [len(search_path) for search_path in search_paths]
            reset_idx = (np.array(search_lens) % self.hidden_horizon_len == 0)

            to_reset_lst = reset_idx.astype(np.int32).tolist()

            # 扩展叶子节点并向后传播以更新统计信息
            for idx in range(batch_size):
                # 扩展叶子节点
                leaf_nodes[idx].expand(next_states[idx], next_value_prefixes[idx]
                                       , next_logits[idx],to_reset_lst[idx])
                # 从叶子节点向根节点传播以更新统计信息
                self.back_propagate(search_paths[idx], next_values[idx], value_min_max_lst[idx])

            # 获取最终结果和信息
        search_root_values = np.asarray([root.get_value() for root in roots])
        search_root_policies = []
        for root, value_min_max in zip(roots, value_min_max_lst):
            policy = self.select_action(root,value_min_max.normalize,1)
            search_root_policies.append(policy)
        search_root_policies = np.asarray(search_root_policies)
        search_best_actions = np.asarray([root.selected_children_idx[0] for root in roots])

        if self.verbose:
            self.log('Final Tree:', verbose=1)
            if batch_size == 1:
                roots[0].print([])
            self.log('search root value -> \t\t {} \n'
                     'search root policy -> \t\t {} \n'
                     'search best action -> \t\t {}'
                     ''.format(search_root_values[0], search_root_policies[0], search_best_actions[0]),
                     verbose=1, iteration_end=True)
        return search_root_values, search_root_policies, search_best_actions, mcts_info

    def sigma_transform(self, max_child_visit_count, value):
        return (self.c_visit + max_child_visit_count) * self.c_scale * value

    def get_transformed_completed_Qs(self, node: Node, value_min_max):
        # 获取完成的 Q 值
        completed_Qs = node.get_completed_Q(value_min_max.normalize)
        # 计算转换后的 Q 值
        max_child_visit_count = max([child.visit_count for child in node.children])
        transformed_completed_Qs = self.sigma_transform(max_child_visit_count, completed_Qs)
        self.log('Get transformed completed Q...\n'
                 'completed Qs -> \t\t {} \n'
                 'max visit count of children -> \t {} \n'
                 'transformed completed Qs -> \t {}'
                 ''.format(completed_Qs, max_child_visit_count, transformed_completed_Qs), verbose=4)
        return transformed_completed_Qs

    def select_action(self, node: Node, value_min_max: MinMaxStats, simulation_idx):

        def takeSecond(elem):
            return elem[1]

        if node.is_root():
            if simulation_idx == 0:
                children_priors = node.get_children_priors()
                children_scores = []
                for a in range(node.num_actions):
                    children_scores.append((a, children_priors[a]))
                children_scores.sort(key=takeSecond, reverse=True)
                for a in range(node.num_actions):
                    node.selected_children_idx.append(children_scores[a][0])

            action = self.do_equal_visit(node)
            self.log('action select at root node, equal visit from {} -> {}'.format(node.selected_children_idx, action),
                     verbose=4)
            return action
        else:
            ## 对于非根节点，按另一种方式计算分数
            # 计算改进后的策略
            policy = node.get_policy()
            Q_value = node.get_normalized_Q(value_min_max.normalize)
            Nb = node.get_children_visit_sum()
            Na = node.get_children_visits()
            #Q_value act_num, policy act_num, Nb 1, Na act_num,
            children_scores = Q_value + policy*(np.sqrt(Nb)/(Na+1))*(self.c_base + np.log((Nb + self.c_scale + 1)/self.c_scale))
            
            action = np.argmax(children_scores)

            self.log('action select at non-root node: \n'
                     'improved policy -> \t\t {} \n'
                     'children visits -> \t\t {} \n'
                     'children scores -> \t\t {} \n'
                     'best action -> \t\t\t {} \n'
                     ''.format(policy, Nb, children_scores, action), verbose=4)

            return action

    def back_propagate(self, search_path, leaf_node_value, value_min_max):
        value = leaf_node_value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node:Node = search_path[i]
            node.estimated_value_lst.append(value)
            
            if i == path_len-1 and i != 0:# boundary node
                parent:Node = search_path[i-1]
                G = node.get_reward() + node.discount*value
                parent.Q_value[node.action] = (node.visit_count*node.Q_value[node.action] + G) / (node.visit_count+1)
                value_min_max.update(parent.Q_value[node.action])

            if i != path_len-1 and i != 0:#non boundary and non root
                parent:Node = search_path[i-1]
                parent.Q_value[node.action] = (node.visit_count*node.Q_value[node.action] + parent.get_qsa(node.action)) / (node.visit_count+1)
                value_min_max.update(parent.Q_value[node.action])

            node.visit_count += 1

    def do_equal_visit(self, node: Node):
        min_visit_count = self.num_simulations + 1
        action = -1
        for selected_child_idx in node.selected_children_idx:
            visit_count = node.children[selected_child_idx].visit_count
            if visit_count < min_visit_count:
                action = selected_child_idx
                min_visit_count = visit_count
        assert action >= 0
        return action

# 简单环境的定义
class SimpleEnv:
    def __init__(self, state_dim=4, num_actions=5):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.reset()
        self.counter = 0

    def reset(self, batch_size=1):
        # 环境重置，并返回初始state
        # 假设state是一个简单的向量，这里随机初始化
        self.current_states = torch.randn(batch_size, self.state_dim)
        return self.current_states

    def step(self, actions):
        # 给定actions，返回下一时刻状态。
        # 对于简单测试环境，我们随机生成下一状态，不关心action影响
        batch_size = actions.shape[0]
        next_states = torch.ones(batch_size, self.state_dim)*self.counter
        self.counter += 1
        self.current_states = next_states
        return next_states

# 简单模型的定义
class SimpleModel(nn.Module):
    def __init__(self, state_dim=4, num_actions=5):
        super(SimpleModel, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # 简单的线性层，仅用于展示结构，不作实际含义
        self.fc_state = nn.Linear(state_dim, state_dim)
        self.fc_reward = nn.Linear(state_dim, 1)
        self.fc_value = nn.Linear(state_dim, 1)
        self.fc_policy = nn.Linear(state_dim, num_actions)

    def forward(self, states):
        # 通用前向：这里对state简单变换，用于模拟一些网络计算
        return self.fc_state(states)

    def next(self, states, actions, task=None):
        # 用于在 MCTS 中预测下一状态（隐状态）
        # 简单地将state过一遍fc_state，并加一些噪声
        next_states = self(states) + 1.*torch.ones_like(states)
        return next_states

    def reward(self, states, actions, task=None):
        # 返回预测的奖励，这里返回随机值（或根据state计算的值）
        # 这里将状态过fc_reward生成一个标量奖励
        r = torch.Tensor(states[:,0]+10.*actions)
        return r

    def Q(self, states, actions, task=None, return_type='avg'):
        # 返回Q值，这里简单地根据state计算一个值即可
        # 对于连续/离散不做区分，简单返回一个标量即可
        Q = torch.Tensor(states[:,0]+10.*actions)
        return Q

    def pi(self, states, task=None):
        # 返回策略的相关信息，这里返回 (mean, std, logits) 三元组
        # 对于离散动作，假定mean=logits, std=恒定张量。为了兼容原有代码的pi返回结构：
        # mean, std只在连续环境用，这里可以简单返回0张量
        # logits = self.fc_policy(states)
        logits = torch.Tensor([[i for i in range(self.num_actions)]])
        return torch.zeros_like(logits), torch.ones_like(logits)*0.1, logits

#########################################
# 下面是一个简单的测试用例
if __name__ == "__main__":
    # 创建简单环境和模型
    env = SimpleEnv(state_dim=4, num_actions=5)
    model = SimpleModel(state_dim=4, num_actions=5)
    
    class CFG:
        num_actions = 5
        num_simulations = 10
        num_top_actions = 5
        c_visit = 1.0
        c_scale = 1.0
        c_base = 1.0
        c_init = 1.0
        dirichlet_alpha = 0.3
        explore_frac = 0.25
        discount = 0.99
        value_minmax_delta = 0.01
        value_support = True
        reward_support = True
        value_prefix = True
        mpc_horizon = 3
        env = 'SimpleEnv'   # 仅用于标识用途
        vis = None
        std_magnification = 1.0

    # 创建一个PyMCTS对象
    from base import MCTS
    # 如果 base.py 不存在，可以直接使用MCTS基类上面定义好的MCTS类即可。
    # MCTS类已经在上面代码中定义，因此这里无需重复定义。

    from math_mcts import MinMaxStats, softmax
    # PyMCTS类已在上述代码中定义。

    cfg = CFG()
    mcts = PyMCTS(cfg)

    # 假设batch_size=1
    batch_size = 1
    root_states = env.reset(batch_size=batch_size)
    # root_values 和 root_policy_logits 可以通过模型获得
    root_values = model.Q(root_states, torch.zeros(batch_size,1).long())
    _, _, root_policy_logits = model.pi(root_states)

    # 进行MCTS搜索测试
    search_root_values, search_root_policies, search_best_actions, mcts_info = \
        mcts.search(model, 
                    batch_size, 
                    root_states, 
                    root_values.detach().cpu().numpy(), 
                    root_policy_logits.detach().cpu().numpy(),
                    task=None,
                    verbose=4)

    print("MCTS Search Results:")
    print("Root Values: ", search_root_values)
    print("Root Policies: ", search_root_policies)
    print("Best Actions: ", search_best_actions)
