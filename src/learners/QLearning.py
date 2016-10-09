class QLearning(object):
    """
    QLearningの学習クラス
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
        action_map_qobj = agent.state_map_action_map_qobj.get(agent.current_state, None)

        prev_action_map_qobj = agent.state_map_action_map_qobj[agent.prev_state]
        prev_q_obj = prev_action_map_qobj[agent.last_action]

        # 終端状態の場合、action_map_qobjがNone
        if action_map_qobj:
            max_q_value = max([qobj.q_value for qobj in action_map_qobj.values()])
        else:
            max_q_value = 0
        prev_q_obj.q_value += self.alpha * (agent.last_reward + self.gamma * max_q_value - prev_q_obj.q_value)
        prev_q_obj.n += 1
