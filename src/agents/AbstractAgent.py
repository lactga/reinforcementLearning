from abc import ABCMeta, abstractmethod


class AbstractAgent(metaclass=ABCMeta):
    """
    強化学習のエージェントの抽象クラス
    """

    def __init__(self, explorer, learner, detector=None, effector=None):
        self.available_action_set = set()
        self.current_episode_num = 1
        self.current_state = None
        self.current_available_action_set = None
        self.prev_state = None
        self.last_reward = None
        self.last_action = None
        self.second_last_action = None
        self.cumsum_reward = 0

        self.list_state_history = []
        self.list_action_history = []
        self.list_reward_history = []

        self.explorer = explorer
        self.learner = learner

        if detector is None:
            def detector(x):
                return x
        self.detector = detector

        if effector is None:
            def effector(x):
                return x
        self.effector = effector

        self.q_function_object = {}

    def set_q_function_object(self, q_function_object):
        self.q_function_object = q_function_object

    @abstractmethod
    def observe_available_action_set(self, available_action_set):
        pass

    @abstractmethod
    def observe_state(self, state):
        pass

    @abstractmethod
    def observe_reward(self, reward):
        pass

    @abstractmethod
    def train(self):
        """
        方策、価値観数などの学習(更新)を行う
        :return:
        """
        pass

    @abstractmethod
    def select_action(self):
        """
        可能な行動から行動を選択する
        :return:
        """
        pass

    def reset_episode(self):
        """
        エピソードをリセットする
        :return:
        """
        self.current_state = None
        self.current_available_action_set = None
        self.prev_state = None
        self.last_reward = None
        self.last_action = None
        self.second_last_action = None
        self.cumsum_reward = 0

        self.list_state_history = []
        self.list_action_history = []
        self.list_reward_history = []
