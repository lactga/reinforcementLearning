import numpy as np
from src.main_class.MainClassOnlineSarsa import MainClassOnlineSarsa
from src.environments.MazeEnvironment import MazeEnvironment
from src.environments.Maze import Maze
from src.agents.AgentQTable import AgentQtable
from src.explorers.EpsilonGreedy import EpsilonGreedy
from src.learners.SarsaLambda import SarsaLambda
from src.plots import plot_maze

if __name__ == '__main__':

    # 環境
    maze_list = np.array([
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 1, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 2, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    ])
    maze = Maze(maze_list=maze_list, limit_time=100, goal_reward=10, hole_reward=-10)
    # maze = Maze(maze_list=maze_list, limit_time=100, goal_reward=10, hole_reward=-100)
    environment = MazeEnvironment(maze=maze)

    # エージェント
    explorer = EpsilonGreedy(epsilon=0.1)
    learner = SarsaLambda(alpha=0.1, gamma=0.9, lamb=0.9)
    agent = AgentQtable(explorer=explorer, learner=learner)

    # メイン処理
    main_class = MainClassOnlineSarsa(environment=environment, agent=agent, episode_loop_num=100, main_loop_num=100)
    main_class.run_main_loop()
    plot_maze.plot_maze(main_class)
