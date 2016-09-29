from src.AbstractEnvironment import AbstractEnvironment
from src.utils.Utils import Utils


class TreasureHuntingEnvironment(AbstractEnvironment):
    """
    宝探しの環境クラス
    """
    def __init__(self, p=0.9, reward=10):
        """
        インストラクタ
        :param p: 思い通りの方向に進める確率
        :param reward: 報酬額
        """

        super().__init__()
        self.p = p
        self.reward = reward

        self.status = '入口'
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

    def proceed_with_step(self, action):
        """
        行動に応じてステップを進め、環境を遷移させ、報酬を発生させ、可能な行動を求める
        :param action: エージェントの行動
        :return: None
        """
        self.t += 1

        # 環境遷移
        dic_transition = self.state_transition_table[self.status][action]
        self.status = Utils.Random.choice(*zip(*list(dic_transition.items())))

        # 報酬テーブルから報酬を発生させる
        self.reward = self.reward_table.get(self.status, 0)

        # 報酬が発生したらエピソード終了。その他の場合は可能な行動は変化しない。
        if self.reward:
            self.list_possible_action = []

