# coding: utf-8
from abc import ABCMeta, abstractstaticmethod, abstractmethod


class AbstractEstimateQValue(metaclass=ABCMeta):
    """
    Q値の推定を行うための抽象クラス
    """

    @abstractstaticmethod
    def estimate_q_value(self, reward, q_obj):
        """
        Q値を推定する関数
        :param reward: 得られた報酬
        :param q_obj: 更新前のQ値を含むオブジェクト
        :return: 更新後のQ値
        """
        pass
