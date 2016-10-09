class Sarsa(object):
    """
    Sarsaの学習クラス
    """

    def __init__(self, alpha=0.1, gamma=0.99):
        """
        インストラクタ

        :param float alpha: 学習率(0 < alpha < 1)
        :param float gamma: 割引率(0 < gamma < 1)
        """
        self.alpha = alpha
        self.gamma = gamma

    def train(self, agent):
        """
        学習を行う

        :param agent: エージェント
        """
        if agent.current_state in agent.state_map_action_map_qobj:
            action_map_qobj = agent.state_map_action_map_qobj[agent.current_state]
            q_obj = action_map_qobj[agent.last_action]
            current_q_value = q_obj.q_value
        else:
            current_q_value = 0

        prev_action_map_qobj = agent.state_map_action_map_qobj[agent.prev_state]
        prev_q_obj = prev_action_map_qobj[agent.second_last_action]

        # 終端状態の場合、action_map_qobjがNone
        prev_q_obj.q_value += self.alpha * (agent.last_reward + self.gamma * current_q_value - prev_q_obj.q_value)
        prev_q_obj.n += 1
