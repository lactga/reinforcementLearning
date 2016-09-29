from src.AbstractAgent import AbstractAgent


class AgentSimpleProfitSharing(AbstractAgent):
    """
    シンプルProfitSharingアルゴリズムのエージェントクラス
    """
    def __init__(self, policy_cls, cbid=0.1):
        """
        インストラクタ
        :param policy_cls: ポリシークラスのインスタンス
        :param cbid: bid率
        """
        self.status_map_action_map_qobj = {}
        self.policy_cls = policy_cls
        self.cbid = cbid
        self.episode = 0
        self.list_selected_rule = []

    def reset_episode(self):
        """
        エピソードを新しく始めるため、内部状態をリセットする
        :return:
        """
        self.list_selected_rule = []

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

        # 最初の観測でない、かつ報酬を得た場合、Q値の更新を行う
        if reward and self.list_selected_rule:
            self.train(reward=reward)
            self.list_selected_rule = []

        # 行動を選択する
        action = self.select_action(status=status)
        self.list_selected_rule.append([status, action])
        self.episode += 1
        return action

    def train(self, reward):
        """
        学習を行う
        :param reward: 報酬
        :return:
        """
        for i, rule in enumerate(self.list_selected_rule):
            status, action = rule
            q_obj = self.status_map_action_map_qobj[status][action]

            r = self.reward_function(t=i, episode=self.episode, reward=reward)

            q_obj['q_value'] += self.cbid * (r - q_obj['q_value'])
            q_obj['n'] += 1

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

    @staticmethod
    def reward_function(t, episode, reward):
        """
        報酬関数（時間によらず一定の場合）
        :param t:
        :param episode:
        :param reward:
        :return:
        """
        return reward
