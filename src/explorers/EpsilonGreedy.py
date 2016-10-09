import random
from src.explorers.AbstractExplorer import AbstractExplorer


class EpsilonGreedy(AbstractExplorer):
    """
    ε-グリーディー戦略のポリシー
    """
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, action_map_qobj):
        """
        行動を選択する関数

        :param action_map_qobj: 行動をキーとする、Qテーブルオブジェクト
        :return: 選択された行動
        """
        if random.random() < self.epsilon:
            action = self._select_action_at_random(action_map_qobj)
        else:
            action = self._get_action_with_max_q_value(action_map_qobj)
        return action

    @staticmethod
    def _select_action_at_random(action_map_qobj):
        """
        一様乱数により行動を選択

        :return: 選択された行動
        """
        action = random.choice(tuple(action_map_qobj.keys()))
        return action

    @staticmethod
    def _get_action_with_max_q_value(action_map_qobj):
        """
        Q値が最大となる行動を選択
        Q値が最大となる行動が複数存在する場合は、Q値が最大となる行動の中からランダムに行動を選択する

        :return: 選択された行動
        """
        list_max_action = []
        max_q_value = float('-inf')

        for action, qobj in action_map_qobj.items():
            if qobj.q_value > max_q_value:
                list_max_action = [action]
                max_q_value = qobj.q_value
            elif qobj.q_value == max_q_value:
                list_max_action.append(action)

        if not list_max_action:
            raise

        max_action = random.choice(list_max_action)
        return max_action

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


