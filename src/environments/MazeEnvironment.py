import numpy as np
from src.environments.AbstractEnvironment import AbstractEnvironment
from src.utils.Utils import Utils


class MazeEnvironment(AbstractEnvironment):
    """
    迷路の環境クラス
    """
    def __init__(self, maze):
        """
        インストラクタ

        :param <maze> maze: 迷路オブジェクト
        """

        super().__init__()
        self.maze = maze

        # 初期状態
        self.current_step_num = 0
        self.current_reward = 0
        self.current_state = self.maze.start
        self.current_available_action_set = set([(1, 0), (-1, 0), (0, 1), (0, -1)])

        # 終了状態の集合
        self.set_end_state = self.maze.endcells

    def perform_action(self, action):
        """
        行動に応じてステップを進め、環境を遷移させ、報酬を発生させ、可能な行動を求める

        :param action: エージェントの行動
        """
        self.current_step_num += 1

        # 状態遷移
        tmp_state = tuple(np.array(self.current_state) + np.array(action))
        if self.maze.get_cell(tmp_state) != self.maze.WALL:  # 移動先が壁の場合は状態遷移しない
            self.current_state = tmp_state

        # 報酬テーブルから報酬を発生させる
        self.current_reward = self.maze.get_reward(self.current_state)
        if self.current_step_num > self.maze.limit_time:
            self.current_reward = self.maze.timeup_reward

        # 終了状態またはlimit_timeであればエピソード終了となり、可能な行動が空となる。その他の場合は可能な行動は変化しない。
        if (self.current_state in self.maze.endcells) or (self.current_step_num > self.maze.limit_time):
            self.current_available_action_set = []

    def reset(self):
        """
        初期状態に戻す
        """
        self.current_step_num = 0
        self.current_reward = 0
        self.current_state = self.maze.start
        self.current_available_action_set = set([(1, 0), (-1, 0), (0, 1), (0, -1)])
