import math

from src.explorers.AbstractExplorer import AbstractExplorer
from src.utils.Utils import Utils


class PolicySoftMax(AbstractExplorer):
    """
    ソフトマックス戦略のポリシー
    """

    def __init__(self, t):
        """
        コンストラクタ
        :param t: 温度パラメタ
        """
        self.t = t

    def select_action(self, action_map_qobj):
        """
        ソフトマックスが最大となる行動を選択
        ソフトマックスが最大となる行動が複数存在する場合は、ソフトマックスが最大となる行動の中からランダムに行動を選択する
        :return: 行動
        """
        list_action = [0] * len(action_map_qobj)
        list_softmax = [0] * len(action_map_qobj)

        for i, items in enumerate(action_map_qobj.items()):
            action, qobj = items
            q_value = qobj['q_value']
            list_action[i] = action
            list_softmax[i] = math.exp(q_value / self.t)

        action = Utils.Random.choice(items=list_action, weights=list_softmax)
        return action




