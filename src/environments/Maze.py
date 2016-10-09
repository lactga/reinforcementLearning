class Maze(object):
    """
    迷路オブジェクト
    """
    START = 1
    GOAL = 2
    HOLE = 8
    WALL = 9

    def __init__(self, maze_list, limit_time=float('inf') , goal_reward=10, hole_reward=-10, timeup_reward=-1):
        """
        インストラクタ
        """
        self.maze_list = maze_list
        self.start = self.search_start()
        self.endcells = self.search_endcells()
        self.reward_table = {
            self.GOAL: goal_reward,
            self.HOLE: hole_reward,
        }
        self.limit_time = limit_time
        self.timeup_reward = timeup_reward

    def search_start(self):
        """
        スタートの場所を取得（ひとつでない場合はエラー）

        :return: スタートの座標
        """
        list_start = []
        for x, maze_line in enumerate(self.maze_list):
            for y, cell in enumerate(maze_line):
                if cell == self.START:
                    list_start.append((x, y))

        if len(list_start) == 1:
            return list_start[0]
        else:
            raise Exception

    def search_endcells(self):
        """
        終了状態の場所を取得（一つもない場合はエラー）

        :return: スタートの座標
        """
        list_goal = []
        for x, maze_line in enumerate(self.maze_list):
            for y, cell in enumerate(maze_line):
                if cell in (self.GOAL, self.HOLE):
                    list_goal.append((x, y))

        if len(list_goal) > 0:
            return list_goal
        else:
            raise Exception

    def get_cell(self, xy):
        return self.maze_list[xy[0]][xy[1]]

    def get_reward(self, xy):
        cell = self.get_cell(xy)
        return self.reward_table.get(cell, 0)
