from src.agents.AbstractAgent import AbstractAgent
from src.learners.QObject import QObject

class AgentQtable(AbstractAgent):
    """
    Qテーブルを用いた学習アルゴリズムのエージェントクラス
    """
    def __init__(self, explorer, learner, detector=None, effector=None):
        """
        インストラクタ

        :param <explorer> explorer: 探索手法クラスのインスタンス
        :param <learner> learner: 学習手法クラスのインスタンス
        :param detector: 受容器関数
        :param effector: 効果器関数
        """
        super().__init__(explorer=explorer, learner=learner, detector=detector, effector=effector)
        self.state_map_action_map_qobj = {}

    def observe_state(self, state):
        """
        状態を観測する

        :param state: 状態
        """
        self.prev_state = self.current_state
        self.current_state = self.detector(state)
        self.list_state_history.append(self.current_state)

    def observe_available_action_set(self, available_action_set):
        """
        可能な行動の集合を観測し、Q値がなければ初期値を用意する

        :param set available_action_set: 可能な行動の集合
        """
        self.available_action_set = available_action_set

        # 該当のQ値がない場合、Q値の初期値を用意する
        if self.current_state not in self.state_map_action_map_qobj:
            for action in available_action_set:
                self._init_qobj(state=self.current_state, action=action)
        else:
            action_map_qobj = self.state_map_action_map_qobj[self.current_state]
            for action in available_action_set:
                if action not in action_map_qobj:
                    self._init_qobj(state=self.current_state, action=action)

    def observe_reward(self, reward):
        """
        報酬を観測する

        :param num reward: 報酬
        """
        self.last_reward = reward
        self.cumsum_reward += reward
        self.list_reward_history.append(reward)

    def select_action(self):
        """
        可能な行動から行動を選択する

        :return: 選択された行動
        """
        action_map_qobj = self.state_map_action_map_qobj[self.current_state]
        action = self.explorer.select_action(action_map_qobj=action_map_qobj)
        self.second_last_action = self.last_action
        self.last_action = self.effector(action)

        self.list_action_history.append(self.last_action)
        return self.last_action

    def train(self):
        """
        Q値の更新を行う
        """
        self.learner.train(agent=self)

    def _init_qobj(self, state, action):
        """
        Qオブジェクトのstate, actionの値を初期化する

        :param state: 状態
        :param action: 行動
        """
        self.state_map_action_map_qobj.setdefault(state, {})
        action_map_qobj = self.state_map_action_map_qobj[state]
        action_map_qobj[action] = QObject(q_value=0, n=0)

    def reset_learned_parameters(self):
        self.state_map_action_map_qobj = {}

    def reset_episode(self):
        super().reset_episode()

        # すべてのs, aに対して。
        for state, action_map_qobj in self.state_map_action_map_qobj.items():
            for action, qobj in action_map_qobj.items():
                # e(s, a)の初期化
                qobj.e = 0