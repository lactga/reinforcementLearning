from abc import ABCMeta, abstractmethod


class AbstractEnvironment(metaclass=ABCMeta):
    """
    強化学習の環境の抽象クラス
    """
    def __init__(self):
        """
        インストラクタ
        """
        self.t = 0
        self.reward = 0
        self.status = None
        self.list_possible_action = []

    @abstractmethod
    def proceed_with_step(self, action):
        """
        行動に応じてステップを進め、環境を遷移させ、報酬を発生させ、可能な行動を求める
        :return:
        """
        pass

    def get_list_possible_action(self):
        """
        可能な行動のリストを取得する
        :return:
        """
        return self.list_possible_action

    def get_status(self):
        """
        状態を取得する
        :return:
        """
        return self.status

    def get_reward(self):
        """
        報酬を取得する
        :return:
        """
        return self.reward

