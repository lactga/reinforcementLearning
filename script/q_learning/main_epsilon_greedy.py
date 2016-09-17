import pickle
import random

import numpy as np

from src.q_learning.Environment import Environment
from src.multi_armed_bandit.PolicyEpsilonGreedy import PolicyEpsilonGreedy
from src.q_learning.Agent import Agent


class Main:
    def __init__(self, episode_num, seed, epsilon):
        """
        初期化

        エージェント、環境を生成
        :param epsilon:
        """
        if seed:
            random.seed(seed)

        self.episode_num = episode_num
        self.agent = Agent(policy_cls=PolicyEpsilonGreedy(epsilon=epsilon), alpha=0.1, gamma=0.99)
        self.environment = Environment()
        self.array_reward = np.zeros(self.episode_num)

    def do_1_episode(self, episode_i):
        # 初期状態を受け取る
        status = self.environment.get_status()
        reward = self.environment.get_reward()
        set_available_action = set(self.environment.get_list_possible_action())

        while True:
            action = self.agent.observe(status=status, reward=reward, set_available_action=set_available_action)
            if action is None:
                break
            self.environment.proceed_with_step(action=action)
            status = self.environment.get_status()
            reward = self.environment.get_reward()
            set_available_action = set(self.environment.get_list_possible_action())
            self.array_reward[episode_i] += reward

    def main(self):
        for i in range(self.episode_num):
            print(i)
            self.environment.__init__()
            # TODO: ちゃんと初期化
            self.agent.prev_status = None
            self.agent.prev_action = None
            self.do_1_episode(episode_i=i)

        for status, action_map_qobj in self.agent.status_map_action_map_qobj.items():
            print(status, action_map_qobj)
        print(tuple(self.array_reward))

if __name__ == '__main__':
    main_cls = Main(episode_num=2000, seed=0, epsilon=0.01)
    main_cls.main()
