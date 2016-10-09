from abc import ABCMeta, abstractmethod


class AbstractAgent(metaclass=ABCMeta):
    """
    強化学習のエージェントの抽象クラス
    """

    def __init__(self, explorer, learner, detector=None, effector=None):
        """
        インストラクタ

        :param <explorer> explorer: 探索手法クラスのインスタンス
        :param <learner> learner: 学習手法クラスのインスタンス
        :param detector: 受容器関数
        :param effector: 効果器関数
        """
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

    @abstractmethod
    def observe_available_action_set(self, available_action_set):
        """
        可能な行動の集合を観測する

        :param set available_action_set: 可能な行動の集合
        """
        pass

    @abstractmethod
    def observe_state(self, state):
        """
        状態を観測する

        :param state: 状態
        """
        pass

    @abstractmethod
    def observe_reward(self, reward):
        """
        報酬を観測する

        :param num reward: 報酬
        """
        pass

    @abstractmethod
    def train(self):
        """
        方策、価値観数などの学習(更新)を行う
        """
        pass

    @abstractmethod
    def select_action(self):
        """
        可能な行動から行動を選択する

        :return: 選択された行動
        """
        pass

    def reset_episode(self):
        """
        エピソードをリセットする
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

        # エピソードカウントを増やす
        self.current_episode_num += 1
