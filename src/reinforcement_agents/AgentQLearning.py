from src.AbstractAgent import AbstractAgent


class AgentQLearning(AbstractAgent):
    """
    Q学習アルゴリズムのエージェントクラス
    """
    def __init__(self, policy_cls, alpha=0.1, gamma=0.99):
        """
        インストラクタ
        :param policy_cls: ポリシークラスのインスタンス
        :param alpha: 学習率
        """
        self.status_map_action_map_qobj = {}
        self.policy_cls = policy_cls
        self.alpha = alpha
        self.gamma = gamma
        self.recent_action = None
        self.prev_action = None
        self.prev_status = None

    def reset_episode(self):
        """
        エピソードを新しく始めるため、内部状態をリセットする
        :return:
        """
        self.recent_action = None
        self.prev_action = None
        self.prev_status = None

    def observe(self, status, reward, set_available_action):
        """
        状態と報酬と可能な行動の集合を観測する
        :param status: 状態
        :param reward: 報酬
        :param set_available_action: 可能な行動の集合
        :return:
        """
        # 現在の状態でのQ値がない場合、Q値の初期値を用意する
        if not status in self.status_map_action_map_qobj:
            self._init_qobj(status=status, set_available_action=set_available_action)

        # 最初の観測でない場合、Q値の更新を行う
        if self.prev_status is not None:
            self.train(status=status, reward=reward)

        # 行動を選択する
        action = self.select_action(status=status)
        self.prev_status = status
        self.prev_action = action
        return action

    def train(self, status, reward):
        """
        学習を行う
        :param status: 状態
        :param reward: 報酬
        :return:
        """
        action_map_qobj = self.status_map_action_map_qobj[status]

        prev_action_map_qobj = self.status_map_action_map_qobj[self.prev_status]
        prev_q_obj = prev_action_map_qobj[self.prev_action]

        # 終端の場合、action_map_qobjが空
        if action_map_qobj:
            max_q_value = max([qobj['q_value'] for qobj in action_map_qobj.values()])
        else:
            max_q_value = 0
        prev_q_obj['q_value'] += self.alpha * (reward + self.gamma * max_q_value - prev_q_obj['q_value'])
        prev_q_obj['n'] += 1

    def select_action(self, status):
        """
        可能な行動から行動を選択する
        :param status: 状態
        :return:
        """
        action_map_qobj = self.status_map_action_map_qobj[status]
        if not action_map_qobj:
            action = None
        else:
            action = self.policy_cls.select_action(action_map_qobj)
        self.recent_action = action
        return action

    def _init_qobj(self, status, set_available_action):
        """
        Qオブジェクトを初期化する
        :param status:
        :param set_available_action:
        :return:
        """
        self.status_map_action_map_qobj[status] = {}
        action_map_qobj = self.status_map_action_map_qobj[status]
        for action in set_available_action:
            action_map_qobj[action] = {'q_value': 0, 'n': 0}
