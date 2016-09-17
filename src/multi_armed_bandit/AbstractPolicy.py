from abc import ABCMeta, abstractstaticmethod


class AbstractPolicy(metaclass=ABCMeta):
    """
    ポリシーの抽象クラス
    """

    @abstractstaticmethod
    def select_action(self, action_map_qobj):
        """
        行動を選択する関数
        :param action_map_qobj: 行動をキーとする、Q値を含むオブジェクト
        :return: 更新後のQ値
        """
        pass
