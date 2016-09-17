import random
from src.multi_armed_bandit.AbstractPolicy import AbstractPolicy


class PolicyGreedy(AbstractPolicy):
    """
    グリーディー戦略のポリシー
    """

    @staticmethod
    def select_action(action_map_qobj):
        """
        行動を選択する関数
        :param action_map_qobj: 行動をキーとする、Q値を含むオブジェクト
        :return: 行動
        """
        action = PolicyGreedy.get_action_with_max_q_value(action_map_qobj)
        return action

    @staticmethod
    def get_action_with_max_q_value(action_map_qobj):
        """
        Q値が最大となる行動を選択
        Q値が最大となる行動が複数存在する場合は、Q値が最大となる行動の中からランダムに行動を選択する
        :return: 行動
        """
        list_max_action = []
        max_q_value = float('-inf')

        for action, qobj in action_map_qobj.items():
            q_value = qobj['q_value']
            if q_value > max_q_value:
                list_max_action = [action]
                max_q_value = q_value
            elif q_value == max_q_value:
                list_max_action.append(action)

        if not list_max_action:
            raise

        max_action = random.choice(list_max_action)
        return max_action




