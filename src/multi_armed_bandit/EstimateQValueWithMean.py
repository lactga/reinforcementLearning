from src.multi_armed_bandit.AbstractEstimateQValue import AbstractEstimateQValue


class EstimateQValueWithMean(AbstractEstimateQValue):
    """
    平均報酬によりQ値の推定を行うクラス
    """
    def __init__(self):
        pass

    @staticmethod
    def estimate_q_value(reward, q_obj):
        """
        平均報酬によりQ値の推定を行う関数
        :param reward: 得られた報酬
        :param q_value: 更新前のQ値を含むオブジェクト
        :return:
        """
        q_value = q_obj['q_value']
        n = q_obj['n']

        new_q_value = (n * q_value + reward) / (n + 1)
        new_n = n + 1

        q_obj['q_value'] = new_q_value
        q_obj['n'] = new_n

