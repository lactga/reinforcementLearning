import random

import numpy as np

from src.multi_armed_bandit.PolicyEpsilonGreedy import PolicyEpsilonGreedy
from src.reinforcement_agents.AgentQLearning import AgentQLearning
from src.reinforcement_environments.HundredGameEnvironment import HundredGameEnvironment


class Main:
    def __init__(self, cycle_num, seed, epsilon, special=0, cards_num=100, list_possible_action=[2, 3, 5]):
        """
        初期化

        エージェント、環境を生成
        :param epsilon:
        """
        if seed:
            random.seed(seed)

        self.cycle_num = cycle_num
        self.agents = [None] * 2
        self.agents[0] = AgentQLearning(policy_cls=PolicyEpsilonGreedy(epsilon=epsilon), alpha=0.1, gamma=0.99)
        self.agents[1] = AgentQLearning(policy_cls=PolicyEpsilonGreedy(epsilon=epsilon), alpha=0.1, gamma=0.99)
        self.environment = None
        self.special = special
        self.cards_num = cards_num
        self.list_possible_action = list_possible_action
        self.array_reward = np.zeros(self.cycle_num)

    def do_1_episode(self, episode_i, is_print_action=False):
        # 初期状態を受け取る
        status = self.environment.get_status()
        reward = self.environment.get_reward()
        action = None
        set_available_action = set(self.environment.get_list_possible_action())

        loop_cnt = episode_i % 2
        while True:
            loop_cnt += 1
            if is_print_action:
                print('prev_action:{}, status:{}, reward:{}, set_available_action={}'.format(
                    action, status, reward, set_available_action))
            action = self.agents[loop_cnt % 2].observe(status=status, reward=reward, set_available_action=set_available_action)
            if action is None:
                break
            self.environment.proceed_with_step(action=action)
            status = self.environment.get_status()
            reward = -1 * self.environment.get_reward()
            set_available_action = set(self.environment.get_list_possible_action())
            if loop_cnt % 2:
                self.array_reward[episode_i] += reward
        # 勝者にも学習をさせる
        loop_cnt += 1
        self.agents[loop_cnt % 2].observe(status=status, reward=-reward, set_available_action=set_available_action)

    def main(self, is_print_loop_counter=True, is_print_Qtable=False, is_print_action=False):
        for i in range(self.cycle_num):
            if is_print_loop_counter:
                print(i)
            self.environment = HundredGameEnvironment(special=self.special, cards_num=self.cards_num,
                                                      list_possible_action=self.list_possible_action)
            for agent in self.agents:
                agent.reset_episode()
            self.do_1_episode(episode_i=i, is_print_action=is_print_action)
            if is_print_Qtable:
                for i in range(2):
                    for status, action_map_qobj in sorted(list(self.agents[i].status_map_action_map_qobj.items()), reverse=True):
                        print(i, status, action_map_qobj)

        for i in range(2):
            for status, action_map_qobj in sorted(list(self.agents[i].status_map_action_map_qobj.items()), reverse=True):
                print(i, status, action_map_qobj)
        print(tuple(self.array_reward))

if __name__ == '__main__':
    cards_num = 21
    list_possible_action = [1, 2, 3]
    main_cls = Main(cycle_num=5, seed=0, epsilon=0.1, special=0, cards_num=cards_num, list_possible_action=list_possible_action)
    main_cls.main(is_print_Qtable=True, is_print_action=True)
    print('='*60)
    main_cls = Main(cycle_num=100000, seed=0, epsilon=0.1, special=0, cards_num=cards_num, list_possible_action=list_possible_action)
    main_cls.main(is_print_loop_counter=False)
    print('=' * 60)
    main_cls = Main(cycle_num=100000, seed=0, epsilon=0.01, special=0, cards_num=cards_num, list_possible_action=list_possible_action)
    main_cls.main(is_print_loop_counter=False)
    print('=' * 60)
    main_cls = Main(cycle_num=100000, seed=0, epsilon=0.01, special=1, cards_num=cards_num, list_possible_action=list_possible_action)
    main_cls.main(is_print_loop_counter=False)


