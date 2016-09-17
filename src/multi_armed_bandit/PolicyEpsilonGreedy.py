import random
from src.multi_armed_bandit.PolicyGreedy import PolicyGreedy


class PolicyEpsilonGreedy(PolicyGreedy):
    """
    ε-グリーディー戦略のポリシー
    """
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, action_map_qobj):
        """
        行動を選択する関数
        :param action_map_qobj: 行動をキーとする、Q値を含むオブジェクト
        :return: 行動
        """
        if random.random() < self.epsilon:
            action = self.select_action_at_random(action_map_qobj)
        else:
            action = self.get_action_with_max_q_value(action_map_qobj)
        return action

    @staticmethod
    def select_action_at_random(action_map_qobj):
        """
        一様乱数により行動を選択
        :return: 行動
        """
        action = random.choice(tuple(action_map_qobj.keys()))
        return action




