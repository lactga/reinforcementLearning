import pickle
import random

import numpy as np

from src.agents.AgentQTable import AgentQLearning
from src.multi_armed_bandit.Environment import Environment
from src.multi_armed_bandit.EstimateQValueWithMean import EstimateQValueWithMean
from src.multi_armed_bandit.PolicySoftMax import PolicySoftMax


def do_1_episode(loop_num, temperature, seed):
    array_reward = np.zeros(loop_num)

    if seed:
        random.seed(seed)
    environment = Environment(n=10)

    # Q関数の初期化
    action_map_qobj = {}
    list_possible_action = environment.get_list_possible_action()
    for action in list_possible_action:
        action_map_qobj[action] = {'q_value': 0, 'n': 0}

    agent = AgentQLearning(estimate_q_value_cls=EstimateQValueWithMean,
                           policy_cls=PolicySoftMax(t=temperature),
                           action_map_qobj=action_map_qobj)

    for i in range(loop_num):
        action = agent.select_action()
        environment.proceed_with_step(action=action)
        reward = environment.get_reward()
        agent.observe(reward)
        array_reward[i] = reward

    return array_reward


def main(temperature=0.1):
    loop_num_each_episode = 1000
    episode_num = 2000
    array_reward = np.zeros(loop_num_each_episode)
    for i in range(episode_num):
        # print(i)
        array_reward += do_1_episode(loop_num=loop_num_each_episode, temperature=temperature, seed=i)

    array_mean_reward = array_reward / episode_num
    print(array_mean_reward)
    with open('out/softmax_{}.dmp'.format(temperature), 'wb') as f:
        pickle.dump(array_mean_reward, file=f)

if __name__ == '__main__':
    main(temperature=1)
    main(temperature=0.1)
    main(temperature=0.01)
