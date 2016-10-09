import numpy as np
import matplotlib.pyplot as plt


class MainClassOnlineQLearning(object):
    """
    オンライン学習型の強化学習のメインクラス
    """

    def __init__(self, environment, agent, episode_loop_num, main_loop_num):
        """
        インストラクタ

        :param <environment> environment: 環境クラスのインスタンス
        :param <agent> agent: エージェントクラスのインスタンス
        :param int episode_loop_num: エピソードの回数
        :param int main_loop_num: メインループの回数
        """
        self.environment = environment
        self.agent = agent
        self.episode_loop_num = episode_loop_num
        self.main_loop_num = main_loop_num

        self.array_reward = np.zeros(self.episode_loop_num)

    def run_main_loop(self):
        """
        強化学習のメインループを実行
        """
        for main_i in range(self.main_loop_num):
            self.agent.reset_learned_parameters()
            for episode_i in range(self.episode_loop_num):
                self.reset_episode()
                self.run_episode()
                if main_i == self.main_loop_num - 1:
                    self.logging(episode=episode_i, is_print=True)
                else:
                    self.logging(episode=episode_i, is_print=False)
            print('main_i:{} '.format(main_i))

        self.array_reward /= self.main_loop_num
        self.print_result()

    def reset_episode(self):
        """
        Episodeのリセット
        """
        self.environment.reset()
        self.agent.reset_episode()

    def run_episode(self):
        """
        エピソードタスクの実行
        """
        # 初期状態を観測
        state = self.environment.get_state()
        self.agent.observe_state(state=state)

        available_action_set = self.environment.get_available_action_set()
        self.agent.observe_available_action_set(available_action_set=available_action_set)

        while True:
            action = self.agent.select_action()
            self.environment.perform_action(action=action)

            state = self.environment.get_state()
            self.agent.observe_state(state=state)

            reward = self.environment.get_reward()
            self.agent.observe_reward(reward=reward)

            available_action_set = self.environment.get_available_action_set()
            self.agent.observe_available_action_set(available_action_set=available_action_set)

            self.agent.train()

            if not available_action_set:
                break

    def logging(self, episode, is_print=True):
        """
        各ステップの結果の保持と出力

        :param int episode: エピソード数
        :param bool is_print: 表示するか
        """
        self.array_reward[episode] += self.agent.cumsum_reward

        if is_print:
            print(self.agent.state_map_action_map_qobj)
            print(list(zip(self.agent.list_state_history, self.agent.list_action_history)), self.agent.cumsum_reward)

    def print_result(self):
        """
        メインループの結果の出力
        """
        for state, action_map_qobj in self.agent.state_map_action_map_qobj.items():
            for action, qobj in action_map_qobj.items():
                print(state, ', ', action, ', ', qobj)

        print(list(self.array_reward))
        print(self.array_reward.mean())

        plt.plot(range(self.episode_loop_num), self.array_reward)
        plt.show()

