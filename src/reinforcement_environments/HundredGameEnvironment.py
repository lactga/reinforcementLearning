import random

from src.environments.AbstractEnvironment import AbstractEnvironment


class HundredGameEnvironment(AbstractEnvironment):
    """
    100ゲームの環境クラス
    """
    REWARD_VALUE = 1

    def __init__(self, cards_num=21, special=0, list_possible_action = [1, 2, 3]):
        """
        インストラクタ
        :param special: スペシャルカードの枚数
        :param reward: 報酬額
        """

        super().__init__()
        self.cards_num = cards_num
        self.special = special
        self.reward = 0
        self.cards = None

        self._init_cards()
        self.list_possible_action = list_possible_action

    def _init_cards(self):
        self.cards = [0] * (self.cards_num - self.special) + [1] * self.special
        random.shuffle(self.cards)

    def proceed_with_step(self, action):
        """
        行動に応じてステップを進め、環境を遷移させ、報酬を発生させ、可能な行動を求める
        :param action: エージェントの行動
        :return: None
        """
        self.t += 1

        # 環境遷移
        self.state_transition(action=action)

        # 報酬テーブルから報酬を発生させる
        if self.cards:
            self.reward = 0
        else:
            self.reward = - self.REWARD_VALUE

        # 報酬が発生したらエピソード終了。その他の場合は可能な行動は変化しない。
        if self.reward:
            self.list_possible_action = []

    def state_transition(self, action):
        """
        actionに従い状態を遷移させる
        :param action:
        :return:
        """
        for i in range(action):
            # カードが引けなければ終了
            if not self.cards:
                return
            card = self.cards.pop()
            # カードがスペシャルなら終了
            if card == 1:
                break

    def get_status(self):
        """
        状態を取得する
        :return:
        """
        return (sum(self.cards), len(self.cards))
