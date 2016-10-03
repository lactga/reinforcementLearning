from src.agents.AbstractAgent import AbstractAgent
from src.learners.QObject import QObject

class AgentQtable(AbstractAgent):
    """
    Qテーブルを用いた学習アルゴリズムのエージェントクラス
    """
    def __init__(self, explorer, learner, detector=None, effector=None):
        super().__init__(explorer=explorer, learner=learner, detector=detector, effector=effector)
        self.state_map_action_map_qobj = {}

    # def __init__(self, policy_cls, alpha=0.1, gamma=0.99):
    #     """
    #     インストラクタ
    #     :param policy_cls: ポリシークラスのインスタンス
    #     :param alpha: 学習率
    #     """

    def observe_state(self, state):
        """
        状態を観測する
        :param state:
        :return:
        """
        self.prev_state = self.current_state
        self.current_state = state
        self.list_state_history.append(state)

    def observe_available_action_set(self, available_action_set):
        """
        可能な行動の集合を観測し、Q値がなければ初期値を用意する
        :param available_action_set:
        :return:
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
        self.last_reward = reward
        self.cumsum_reward += reward
        self.list_reward_history.append(reward)

    def select_action(self):
        """
        可能な行動から行動を選択する
        :return:
        """
        action_map_qobj = self.state_map_action_map_qobj[self.current_state]
        action = self.explorer.select_action(action_map_qobj=action_map_qobj)
        self.second_last_action = self.last_action
        self.last_action = action

        self.list_action_history.append(action)
        return action

    def train(self):
        """
        学習を行う
        :return:
        """
        self.learner.train(agent=self)

    def _init_qobj(self, state, action):
        """
        Qオブジェクトを初期化する
        :param state:
        :param action:
        :return:
        """
        self.state_map_action_map_qobj.setdefault(state, {})
        action_map_qobj = self.state_map_action_map_qobj[state]
        action_map_qobj[action] = QObject(q_value=0, n=0)
