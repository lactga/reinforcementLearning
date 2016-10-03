import numpy as np

class MainClassOnlineSarsa(object):
    """
    オンライン学習型の強化学習のメインクラス
    """

    def __init__(self, environment, agent, main_loop_num):
        self.environment = environment
        self.agent = agent
        self.main_loop_num = main_loop_num

        self.list_reward = []

    def run_main_loop(self):
        """
        強化学習のメインループを実行
        :return:
        """
        for i in range(self.main_loop_num):
            self.reset_episode()
            self.run_episode()
            self.logging()

        self.print_result()

    def reset_episode(self):
        """
        Episodeのリセット
        :return:
        """
        self.environment.reset()
        self.agent.reset_episode()

    def run_episode(self):
        """
        エピソードタスクの実行
        :return:
        """
        # 初期状態を観測
        state = self.environment.get_state()
        self.agent.observe_state(state=state)

        available_action_set = self.environment.get_available_action_set()
        self.agent.observe_available_action_set(available_action_set=available_action_set)

        action = self.agent.select_action()
        self.environment.perform_action(action=action)

        while True:
            state = self.environment.get_state()
            self.agent.observe_state(state=state)

            reward = self.environment.get_reward()
            self.agent.observe_reward(reward=reward)

            available_action_set = self.environment.get_available_action_set()
            self.agent.observe_available_action_set(available_action_set=available_action_set)

            if not available_action_set:
                break

            action = self.agent.select_action()
            self.agent.train()
            self.environment.perform_action(action=action)

        self.agent.train()

    def logging(self):
        """
        結果の保持と出力
        :return:
        """
        self.list_reward.append(self.agent.cumsum_reward)

        print(self.agent.state_map_action_map_qobj)
        print(list(zip(self.agent.list_state_history, self.agent.list_action_history)), self.agent.cumsum_reward)

    def print_result(self):
        print(list(np.cumsum(self.list_reward)))
        print(list(np.cumsum(self.list_reward) / np.arange(1, 1 + self.main_loop_num)))

