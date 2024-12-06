import numpy as np
from math_mcts import *

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
        logits = np.asarray([[child.prior] for child in self.children])
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
        #process numpy to float
        value_prefix = self.value_prefix.item() if not isinstance(self.value_prefix,float)else self.value_prefix
        reward = self.get_reward()
        reward = reward.item() if not isinstance(reward,float) else reward
        value = self.get_value()
        value = value.item() if not isinstance(value,float) else value

        s = '[a={} reset={} (n={}, vp={:.3f} r={:.3f}, v={:.3f})]'.format(
            action, self.reset_value_prefix, self.visit_count, value_prefix, reward, value)
        return s