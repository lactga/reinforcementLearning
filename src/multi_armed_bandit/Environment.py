from src.AbstractEnvironment import AbstractEnvironment
import random


class Environment(AbstractEnvironment):
    """
    多腕バンディット問題の環境クラス
    """
    def __init__(self, n):
        """
        インストラクタ
        """
        super().__init__()
        self.list_possible_action = tuple(range(n))

        self.list_q = [0] * n
        for i in range(n):
            self.list_q[i] = random.gauss(mu=0, sigma=1)

    def proceed_with_step(self, action):
        """
        行動に応じてステップを進め、環境を遷移させ、報酬を発生させ、可能な行動を求める
        :return:
        """
        self.t += 1
        self.reward = random.gauss(mu=self.list_q[action], sigma=1)

