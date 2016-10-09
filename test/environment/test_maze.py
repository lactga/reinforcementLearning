from unittest import TestCase
from src.environments.Maze import Maze


class TestMaze(TestCase):

    def test_search_start_1(self):
        maze_list = [
            [9, 9, 9, 9, 9],
            [9, 2, 0, 0, 9],
            [9, 0, 0, 1, 9],
            [9, 9, 9, 9, 9],
        ]

        maze = Maze(maze_list=maze_list)
        start = maze.search_start()
        self.assertEqual(start, (2, 3))

    def test_search_start_2(self):
        maze_list = [
            [9, 9, 9, 9, 9],
            [9, 2, 1, 0, 9],
            [9, 0, 0, 1, 9],
            [9, 9, 9, 9, 9],
        ]

        with self.assertRaises(Exception):
            Maze(maze_list=maze_list)

    def test_search_goal_1(self):
        maze_list = [
            [9, 9, 9, 9, 9],
            [9, 0, 2, 0, 9],
            [9, 0, 0, 1, 9],
            [9, 9, 9, 9, 9],
        ]

        maze = Maze(maze_list=maze_list)
        goal = maze.search_endcells()
        self.assertEqual(goal, [(1, 2)])

    def test_search_goal_2(self):
        maze_list = [
            [9, 9, 9, 9, 9],
            [9, 2, 2, 2, 9],
            [9, 0, 0, 1, 9],
            [9, 9, 9, 9, 9],
        ]

        maze = Maze(maze_list=maze_list)
        goal = maze.search_endcells()
        self.assertEqual(goal, [(1, 1), (1, 2), (1, 3)])

    def test_search_goal_3(self):
        maze_list = [
            [9, 9, 9, 9, 9],
            [9, 0, 0, 0, 9],
            [9, 0, 0, 1, 9],
            [9, 9, 9, 9, 9],
        ]

        with self.assertRaises(Exception):
            Maze(maze_list=maze_list)
