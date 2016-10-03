import random

import numpy as np

from src.agents.AgentQTable import AgentQLearning
from src.environments.TreasureHuntingEnvironment import TreasureHuntingEnvironment
from src.explorers.EpsilonGreedy import EpsilonGreedy


class Main:
    def __init__(self, cycle_num, seed, epsilon, reward=10, p=0.9):
        """
        初期化

        エージェント、環境を生成
        :param epsilon:
        """
        if seed:
            random.seed(seed)

        self.cycle_num = cycle_num
        self.agent = AgentQLearning(policy_cls=EpsilonGreedy(epsilon=epsilon), alpha=0.1, gamma=0.99)
        self.environment = None
        self.reward = reward
        self.p = p
        self.array_reward = np.zeros(self.cycle_num)

    def do_1_episode(self, episode_i, is_print_action=False):
        # 初期状態を受け取る
        status = self.environment.get_status()
        reward = self.environment.get_reward()
        action = None
        available_action_set = set(self.environment.get_list_possible_action())

        while True:
            if is_print_action:
                print('prev_action:{}, status:{}, reward:{}, available_action_set={}'.format(
                    action, status, reward, available_action_set))
            action = self.agent.observe(status=status, reward=reward, available_action_set=available_action_set)
            if action is None:
                break
            self.environment.proceed_with_step(action=action)
            status = self.environment.get_status()
            reward = self.environment.get_reward()
            available_action_set = set(self.environment.get_list_possible_action())
            self.array_reward[episode_i] += reward

    def main(self, is_print_loop_counter=True, is_print_Qtable=False, is_print_action=False):
        for i in range(self.cycle_num):
            if is_print_loop_counter:
                print(i)
            self.environment = TreasureHuntingEnvironment(p=self.p, reward=self.reward)
            self.agent.reset_episode()
            self.do_1_episode(episode_i=i, is_print_action=is_print_action)
            if is_print_Qtable:
                for status, action_map_qobj in self.agent.status_map_action_map_qobj.items():
                    print(status, action_map_qobj)

        for status, action_map_qobj in self.agent.status_map_action_map_qobj.items():
            print(status, action_map_qobj)
        print(tuple(self.array_reward))

if __name__ == '__main__':
    main_cls = Main(cycle_num=5, seed=0, epsilon=0.01, p=1, reward=10)
    main_cls.main(is_print_Qtable=True, is_print_action=True)
    print('='*60)
    main_cls = Main(cycle_num=2000, seed=0, epsilon=0.01, p=1, reward=10)
    main_cls.main(is_print_loop_counter=False)
    print('=' * 60)
    main_cls = Main(cycle_num=2000, seed=0, epsilon=0.01, p=0.6, reward=10)
    main_cls.main(is_print_loop_counter=False)

