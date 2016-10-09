from abc import ABCMeta, abstractmethod


class AbstractEnvironment(metaclass=ABCMeta):
    """
    強化学習の環境の抽象クラス
    """
    def __init__(self):
        """
        インストラクタ
        """
        self.current_step_num = 0
        self.current_reward = 0
        self.current_state = None
        self.current_available_action_set = []

    @abstractmethod
    def perform_action(self, action):
        """
        行動に応じてステップを進め、環境を遷移させ、報酬を発生させ、可能な行動を求める

        :param action: 行動
        """
        pass

    @abstractmethod
    def reset(self):
        """
        初期状態に戻す
        """

    def get_available_action_set(self):
        return self.current_available_action_set

    def get_state(self):
        return self.current_state

    def get_reward(self):
        return self.current_reward

    def get_step_num(self):
        return self.current_step_num

