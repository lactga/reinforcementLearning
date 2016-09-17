import pickle
import random

import numpy as np

from src.multi_armed_bandit.Environment import Environment
from src.multi_armed_bandit.EstimateQValueWithMean import EstimateQValueWithMean
from src.multi_armed_bandit.PolicyGreedy import PolicyGreedy
from src.q_learning.Agent import Agent


def do_1_episode(loop_num, seed=None):
    array_reward = np.zeros(loop_num)

    if seed:
        random.seed(seed)
    environment = Environment(n=10)

    # Q関数の初期化
    action_map_qobj = {}
    list_possible_action = environment.get_list_possible_action()
    for action in list_possible_action:
        action_map_qobj[action] = {'q_value': 0, 'n': 0}

    agent = Agent(estimate_q_value_cls=EstimateQValueWithMean,
                  policy_cls=PolicyGreedy,
                  action_map_qobj=action_map_qobj)

    for i in range(loop_num):
        action = agent.select_action()
        environment.proceed_with_step(action=action)
        reward = environment.get_reward()
        agent.observe(reward)
        array_reward[i] = reward

    return array_reward


def main():
    loop_num_each_episode = 1000
    episode_num = 2000
    array_reward = np.zeros(loop_num_each_episode)
    for i in range(episode_num):
        print(i)
        array_reward += do_1_episode(loop_num=loop_num_each_episode, seed=i)

    array_mean_reward = array_reward / episode_num
    print(array_mean_reward)
    with open('out/greedy.dmp', 'wb') as f:
        pickle.dump(array_mean_reward, file=f)

if __name__ == '__main__':
    main()
