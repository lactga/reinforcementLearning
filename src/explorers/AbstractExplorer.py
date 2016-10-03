from abc import ABCMeta, abstractstaticmethod


class AbstractExplorer(metaclass=ABCMeta):
    """
    行動選択手法の抽象クラス
    """

    @abstractstaticmethod
    def select_action(self, action_map_qobj):
        """
        行動を選択する関数
        :param action_map_qobj: 行動をキーとする、Qテーブルオブジェクト
        :return: 更新後のQ値
        """
        pass
