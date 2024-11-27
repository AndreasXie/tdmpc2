import copy 
import torch
from torch.cuda.amp import autocast as autocast
import numpy as np
from torch import nn
import math
import torch.nn.functional as F
from torch.distributions import TransformedDistribution, Normal, constraints
from torch.distributions import transforms
from .base import MCTS

device = "cpu"
# 定义 softmax 函数
def softmax(logits):
    logits = np.asarray(logits)
    logits -= logits.max()
    e_x = np.exp(logits)
    return e_x / e_x.sum()

# 定义 TanhTransform
class TanhTransform(transforms.Transform):
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

# 定义 SquashedNormal 分布
class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        transforms_ = [TanhTransform()]
        super().__init__(self.base_dist, transforms_)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

# 定义 MinMaxStats 类
class MinMaxStats:
    def __init__(self, minmax_delta, min_value_bound=None, max_value_bound=None):
        """
        Minimum and Maximum statistics
        :param minmax_delta: float, for soft update
        :param min_value_bound:
        :param max_value_bound:
        """
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')
        self.minmax_delta = minmax_delta

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            if value >= self.maximum:
                value = self.maximum
            elif value <= self.minimum:
                value = self.minimum
            # 仅在设置了最大值和最小值时进行归一化
            value = (value - self.minimum) / max(self.maximum - self.minimum, self.minmax_delta)  # [0, 1] 范围

        value = max(min(value, 1), 0)
        return value

    def clear(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

# 定义 Node 类
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

    def get_policy(self):
        logits = np.asarray([child.prior for child in self.children])
        return softmax(logits)

    def get_improved_policy(self, transformed_completed_Qs):
        logits = np.asarray([child.prior for child in self.children])
        return softmax(logits + transformed_completed_Qs)

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

        s = '[a={} reset={} (n={}, vp={:.3f} r={:.3f}, v={:.3f})]'.format(
            action, self.reset_value_prefix, self.visit_count, self.value_prefix, self.get_reward(), self.get_value())
        return s


# 继承自 MCTS 基类的 PyMCTS 类
class PyMCTS(MCTS):
    def __init__(self, cfg):
        super().__init__(cfg)

    def sample_actions(self, policy, add_noise=True, temperature=1.0, input_noises=None, input_dist=None, input_actions=None):
        batch_size = policy.shape[0]
        n_policy = self.num_top_actions
        n_random = self.num_top_actions  # 假设随机动作数等于顶级动作数
        std_magnification = self.std_magnification
        action_dim = policy.shape[-1] // 2

        if input_dist is not None:
            n_policy //= 2
            n_random //= 2

        Dist = SquashedNormal
        mean, std = policy[:, :action_dim], policy[:, action_dim:]
        distr = Dist(mean, std)
        sampled_actions = distr.sample(torch.Size([n_policy + n_random]))
        sampled_actions = sampled_actions.permute(1, 0, 2)

        policy_actions = sampled_actions[:, :n_policy]
        random_actions = sampled_actions[:, -n_random:]

        if add_noise:
            if input_noises is None:
                # random_distr = Dist(mean, self.std_magnification * std * temperature)       # more flatten gaussian policy
                random_distr = Dist(mean, std_magnification * std)  # more flatten gaussian policy
                random_actions = random_distr.sample(torch.Size([n_random]))
                random_actions = random_actions.permute(1, 0, 2)

                # random_actions = torch.rand(batch_size, n_random, action_dim).float().cuda()
                # random_actions = 2 * random_actions - 1

                # Gaussian noise
                # random_actions += torch.randn_like(random_actions)
            else:
                noises = torch.from_numpy(input_noises).float().to(device)
                random_actions += noises

        if input_dist is not None:
            refined_mean, refined_std = input_dist[:, :action_dim], input_dist[:, action_dim:]
            refined_distr = Dist(refined_mean, refined_std)
            refined_actions = refined_distr.sample(torch.Size([n_policy + n_random]))
            refined_actions = refined_actions.permute(1, 0, 2)

            refined_policy_actions = refined_actions[:, :n_policy]
            refined_random_actions = refined_actions[:, -n_random:]

            if add_noise:
                if input_noises is None:
                    refined_random_distr = Dist(refined_mean, std_magnification * refined_std)
                    refined_random_actions = refined_random_distr.sample(torch.Size([n_random]))
                    refined_random_actions = refined_random_actions.permute(1, 0, 2)
                else:
                    noises = torch.from_numpy(input_noises).float().to(device)
                    refined_random_actions += noises

        all_actions = torch.cat((policy_actions, random_actions), dim=1)
        if input_actions is not None:
            all_actions = torch.from_numpy(input_actions).float().to(device)
        if input_dist is not None:
            all_actions = torch.cat((all_actions, refined_policy_actions, refined_random_actions), dim=1)
        # all_actions[:, 0, :] = mean     # add mean as one of candidate
        all_actions = all_actions.clip(-0.999, 0.999)

        # probs = distr.log_prob(all_actions.permute(1, 0, 2)).exp().mean(-1).permute(1, 0)
        probs = None
        return all_actions, probs

    def search(self, model, batch_size, root_states, root_values, root_policy_logits, task, 
               use_gumble_noise=True, temperature=1.0, verbose=0, **kwargs):
        # 准备工作
        Node.set_static_attributes(self.discount, self.num_actions)  # 设置 MCTS 的静态参数
        roots = [Node(prior=1) for _ in range(batch_size)]          # 为批次设置根节点
        # 扩展根节点并更新统计信息
        for root, state, value, logit in zip(roots, root_states, root_values, root_policy_logits):
            root.expand(state, 0, logit)
            root.estimated_value_lst.append(value)
            root.visit_count += 1
        # 保存树节点的最小和最大值
        value_min_max_lst = [MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]

        # 设置 Gumble 噪声（在训练期间）
        if use_gumble_noise:
            gumble_noises = np.random.gumbel(0, 1, (batch_size, self.num_actions)) * temperature
        else:
            gumble_noises = np.zeros((batch_size, self.num_actions))

        assert batch_size == len(root_states) == len(root_values)
        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(precision=3)
            # assert batch_size == 1  # 取消断言，允许 batch_size >1

            self.log('Gumble Noise:\n{}'.format(gumble_noises), verbose=1)

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
                    action = self.select_action(node, value_min_max, gumble_noises[idx], simulation_idx)
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
                self.back_propagate(search_paths[idx], next_values[idx].item(), value_min_max_lst[idx])

            if self.ready_for_next_gumble_phase(simulation_idx):
                # 最终选择
                for idx in range(batch_size):
                    root, gumble_noise, value_min_max = roots[idx], gumble_noises[idx], value_min_max_lst[idx]
                    self.sequential_halving(root, gumble_noise, value_min_max)
                self.log('change to phase: {}, top m action -> {}'.format(
                    self.current_phase, self.current_num_top_actions), verbose=3)

            # 获取最终结果和信息
        search_root_values = np.asarray([root.get_value() for root in roots])
        search_root_policies = []
        for root, value_min_max in zip(roots, value_min_max_lst):
            improved_policy = root.get_improved_policy(self.get_transformed_completed_Qs(root, value_min_max))
            search_root_policies.append(improved_policy)
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

    def select_action(self, node: Node, value_min_max: MinMaxStats, gumbel_noise, simulation_idx):

        def takeSecond(elem):
            return elem[1]

        if node.is_root():
            if simulation_idx == 0:
                children_priors = node.get_children_priors()
                children_scores = []
                for a in range(node.num_actions):
                    children_scores.append((a, gumbel_noise[a] + children_priors[a]))
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
            improved_policy = node.get_improved_policy(self.get_transformed_completed_Qs(node, value_min_max))
            children_visits = node.get_children_visits()
            # 计算每个子节点的分数
            children_scores = [improved_policy[action] - children_visits[action] / (1 + node.get_children_visit_sum())
                               for action in range(node.num_actions)]
            action = np.argmax(children_scores)
            self.log('action select at non-root node: \n'
                     'improved policy -> \t\t {} \n'
                     'children visits -> \t\t {} \n'
                     'children scores -> \t\t {} \n'
                     'best action -> \t\t\t {} \n'
                     ''.format(improved_policy, children_visits, children_scores, action), verbose=4)
            return action

    def back_propagate(self, search_path, leaf_node_value, value_min_max):
        value = leaf_node_value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            node.estimated_value_lst.append(value)
            node.visit_count += 1

            value = node.get_reward() + self.discount * value
            self.log('Update min max value [{:.3f}, {:.3f}] by {:.3f}'.format(
                value_min_max.minimum, value_min_max.maximum, value), verbose=3)
            value_min_max.update(value)

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

    def ready_for_next_gumble_phase(self, simulation_idx):
        ready = (simulation_idx + 1) >= self.visit_num_for_next_phase
        if ready:
            self.current_phase += 1
            self.current_num_top_actions //= 2
            assert self.current_num_top_actions == self.num_top_actions // (2 ** self.current_phase)

            # 更新下一个阶段的总访问次数
            n = self.num_simulations
            m = self.num_top_actions
            current_m = self.current_num_top_actions
            # 当前阶段的访问次数
            if current_m > 2:
                extra_visit = np.floor(n / (np.log2(m) * current_m)) * current_m
            else:
                extra_visit = n - self.used_visit_num
            self.used_visit_num += extra_visit
            self.visit_num_for_next_phase += extra_visit
            self.visit_num_for_next_phase = min(self.visit_num_for_next_phase, self.num_simulations)

            self.log('Be ready for the next gumble phase at iteration {}: \n'
                     'current top action num is {}, visit {} times for next phase'.format(
                         simulation_idx, current_m, self.visit_num_for_next_phase), verbose=3)
        return ready

    def sequential_halving(self, root, gumble_noise, value_min_max):
        ## 更新根节点的当前选择的顶级 m 个动作
        children_prior = root.get_children_priors()
        if self.current_phase == 0:
            # 第一阶段：分数 = g + logits 来自所有子节点
            children_scores = np.asarray([gumble_noise[action] + children_prior[action]
                                          for action in range(root.num_actions)])
            sorted_action_index = np.argsort(children_scores)[::-1]  # 从大到小排序分数
            # 获取前 m 个动作
            root.selected_children_idx = sorted_action_index[:self.current_num_top_actions]

            self.log('Do sequential halving at phase {}: \n'
                     'gumble noise -> \t\t {} \n'
                     'child prior -> \t\t\t {} \n'
                     'children scores -> \t\t {} \n'
                     'the selected children indexes -> {}'.format(
                         self.current_phase, gumble_noise, children_prior, children_scores,
                         root.selected_children_idx), verbose=3)
        else:
            assert len(root.selected_children_idx) > 1
            # 后续阶段：分数 = g + logits + sigma(hat_q) 来自选定的子节点
            transformed_completed_Qs = self.get_transformed_completed_Qs(root, value_min_max)
            # 选定的子节点索引
            selected_children_idx = root.selected_children_idx
            children_scores = np.asarray([gumble_noise[action] + children_prior[action] +
                                          transformed_completed_Qs[action]
                                          for action in selected_children_idx])
            sorted_action_index = np.argsort(children_scores)[::-1]  # 从大到小排序分数
            # 选择 top m / 2 个动作
            if isinstance(selected_children_idx,list):
                selected_children_idx = np.array(selected_children_idx)
            root.selected_children_idx = selected_children_idx[sorted_action_index[:self.current_num_top_actions]].tolist()
            self.log('Do sequential halving at phase {}: \n'
                     'selected children -> \t\t {} \n'
                     'gumble noise -> \t\t {} \n'
                     'child prior -> \t\t\t {} \n'
                     'transformed completed Qs -> \t {} \n'
                     'children scores -> \t\t {} \n'
                     'the selected children indexes -> {}'.format(
                         self.current_phase, selected_children_idx, gumble_noise[selected_children_idx],
                         children_prior[selected_children_idx],
                         transformed_completed_Qs[selected_children_idx], children_scores,
                         root.selected_children_idx), verbose=3)

        best_action = root.selected_children_idx[0]
        return best_action

# 实现一个简单的环境类，支持批量操作
class SimpleEnv:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        # 初始化所有批次的状态为 0
        self.states = np.zeros((self.batch_size, 1))
        return self.states

    def step(self, actions):
        """
        :param actions: ndarray of shape (batch_size,)
        :return: next_states, rewards, done, info
        """
        # 动作 0 减少状态，动作 1 增加状态
        actions = actions.astype(int)
        self.states += np.where(actions == 0, -1, 1)
        rewards = self.states.flatten().astype(float)
        done = np.zeros(self.batch_size, dtype=bool)  # 为简化起见，episode 不会结束
        info = {}
        return self.states, rewards, done, info

# 实现一个简单的模型，输出有规律的值，并实现 recurrent_inference 方法
class SimpleModel:
    def __init__(self, num_actions, batch_size):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.iteration = 0  # 用于生成有规律的输出

    def recurrent_inference(self, states, actions):
        """
        生成有规律的输出，以便于调试和验证。
        :param states: Tensor of shape (batch_size, state_dim)
        :param actions: Tensor of shape (batch_size, 1)
        :return: next_states, next_value_prefixes, next_values, next_logits
        """
        self.iteration += 1  # 增加迭代计数

        batch_size = states.size(0)
        state_dim = states.size(1)

        # 生成下一个状态：当前状态 + 动作
        # 动作为 0 或 1，对应 -1 或 +1
        action_effect = torch.where(actions == 0, -1.0, 1.0).to(states.device)
        next_states = states + action_effect.unsqueeze(1)

        # 生成 value_prefix：设置为当前迭代数
        next_value_prefixes = torch.full((batch_size,), self.iteration, device=states.device)

        # 生成 next_values：设置为下一个状态的和（有规律增长）
        next_values = next_states.sum(dim=1)

        # 生成 next_logits：每个动作的 logits 按照自然数排列从小到大
        # 例如，对于 num_actions=2，batch_size=10，logits 为 [[1,2], [2,3], ..., [10,11]]
        batch_indices = torch.arange(1, batch_size + 1, device=states.device).float().unsqueeze(1)
        next_logits = torch.arange(1, self.num_actions + 1, device=states.device).float().unsqueeze(0) + batch_indices

        return next_states, next_value_prefixes, next_values, next_logits
    
class MCTSParams:
    def __init__(self, num_actions, env):
        self.num_simulations = 50          # 模拟次数
        self.num_top_actions = num_actions # 顶级动作数量
        self.c_visit = 1.0                 # 访问计数常数
        self.c_scale = 1.0                 # 缩放常数
        self.c_base = 19652                # 基础常数
        self.c_init = 1.25                 # 初始化常数
        self.dirichlet_alpha = 0.3         # Dirichlet 分布的 alpha 参数
        self.explore_frac = 0.25           # 探索比例
        self.discount = 0.99               # 折扣因子
        self.value_minmax_delta = 1e-6     # 最小最大值更新的 delta
        self.value_support = (-1, 1)       # 值的支持范围
        self.reward_support = (-1, 1)      # 奖励的支持范围
        self.value_prefix = True           # 是否使用值前缀
        self.mpc_horizon = 10              # MPC 的时间步长
        self.env = env                     # 环境实例
        self.vis = ['log', 'text', 'graph']# 可视化选项
        self.std_magnification = 1.0       # 标准差放大倍数
        self.num_actions = num_actions     # 动作数量

# 测试 MCTS 功能
def test_mcts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 初始化参数
    batch_size = 10
    num_actions = 4

    # 初始化环境
    env = SimpleEnv(batch_size=batch_size)

    # 初始化模型
    model = SimpleModel(num_actions=num_actions, batch_size=batch_size)

    # 定义 MCTS 的参数
    mcts_params = MCTSParams(4,'atari')

    # 初始化 MCTS
    mcts = PyMCTS(mcts_params)
    mcts.verbose = 2  # 设置详细级别

    # 设置根状态
    root_state = torch.tensor(env.reset()).float().to(device)  # 初始状态，形状 [batch_size, state_dim]
    root_value = [0.0 for _ in range(batch_size)]  # 初始值
    root_policy_logits = [np.zeros(num_actions) for _ in range(batch_size)]  # 初始策略 logits

    # 运行 MCTS 搜索
    search_root_values, search_root_policies, search_best_actions, mcts_info = mcts.search(
        model=model,
        batch_size=batch_size,
        root_states=root_state,
        root_values=root_value,
        root_policy_logits=root_policy_logits,
        use_gumble_noise=False,
        temperature=1.0,
        verbose=0
    )

    print("Search root values:", search_root_values)
    print("Search root policies:", search_root_policies)
    print("Search best actions:", search_best_actions)

# 运行测试
if __name__ == "__main__":
    test_mcts()
