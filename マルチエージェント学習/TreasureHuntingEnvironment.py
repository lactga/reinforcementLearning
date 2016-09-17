from Utils import Utils


class TreasureHuntingEnvironment():
    """
    宝探しの環境クラス
    """
    def __init__(self, p=0.9, reward=10):
        """
        インストラクタ
        """
        self.p = p
        self.reward = reward

        self.status = '入口'
        self.reward = 0
        self.list_possible_action = ['西', '東']

        self.state_transition_table = {
            '湖': {
                '西': {'山賊': p, '入口': 1 - p},
                '東': {'入口': p, '山賊': 1 - p}
            },
            '入口': {
                '西': {'湖': p, '森': 1 - p},
                '東': {'森': p, '湖': 1 - p}
            },
            '森': {
                '西': {'入口': p, '宝': 1 - p},
                '東': {'宝': p, '入口': 1 - p}
            },
        }
        self.reward_table = {
            '山賊': -reward,
            '宝': reward,
        }

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

    def get_list_possible_action(self):
        """
        可能な行動のリストを取得する
        :return:
        """
        return self.list_possible_action

    def proceed_with_step(self, action):
        """
        行動に応じて環境を遷移させ、報酬を発生させ、可能な行動を求める
        :return:
        """
        dic_transition = self.state_transition_table[self.status][action]
        self.status = Utils.Random.choice(*zip(*list(dic_transition.items())))
        self.reward = self.reward_table.get(self.status, 0)

        # 報酬が発生したらエピソード終了。その他は可能な行動は変化しない。
        if self.reward:
            self.list_possible_action = []

