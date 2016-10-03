from src.environments.AbstractEnvironment import AbstractEnvironment
from src.utils.Utils import Utils


class TreasureHuntingEnvironment(AbstractEnvironment):
    """
    宝探しの環境クラス
    """
    def __init__(self, p=0.9, reward_value=10):
        """
        インストラクタ
        :param p: 思い通りの方向に進める確率
        :param reward: 報酬額
        """

        super().__init__()
        # 初期状態
        self.current_step_num = 0
        self.current_reward = 0
        self.current_state = '入口'
        self.current_available_action_set = ['西', '東']

        # 引数を保持(確認用)
        self.p = p
        self.reward_value = reward_value

        # 状態遷移テーブル
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

        # 報酬テーブル
        self.reward_table = {
            '山賊': -reward_value,
            '宝': reward_value,
        }

        # 終了状態の集合
        self.set_end_state = {'山賊', '宝'}

    def perform_action(self, action):
        """
        行動に応じてステップを進め、環境を遷移させ、報酬を発生させ、可能な行動を求める
        :param action: エージェントの行動
        :return: None
        """
        self.current_step_num += 1

        # 環境遷移
        dic_transition = self.state_transition_table[self.current_state][action]
        self.current_state = Utils.Random.choice(*zip(*list(dic_transition.items())))

        # 報酬テーブルから報酬を発生させる
        self.current_reward = self.reward_table.get(self.current_state, 0)

        # 終了状態であればエピソード終了となり、可能な行動が空となる。その他の場合は可能な行動は変化しない。
        if self.current_state in self.set_end_state:
            self.current_available_action_set = []

    def reset(self):
        """
        初期状態に戻す
        :return:
        """
        self.current_step_num = 0
        self.current_reward = 0
        self.current_state = '入口'
        self.current_available_action_set = ['西', '東']
