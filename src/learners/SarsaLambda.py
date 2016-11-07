from src.learners.QObject import QObject

class SarsaLambda(object):
    """
    Sarsa(λ)の学習クラス
    """

    def __init__(self, alpha=0.1, gamma=0.99, lamb=0.9):
        """
        インストラクタ

        :param float alpha: 学習率(0 < alpha < 1)
        :param float gamma: 割引率(0 < gamma < 1)
        :param float lamb: λ(0 <= lamd <= 1)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb

    def train(self, agent):
        """
        学習を行う

        :param agent: エージェント
        """

        #
        if agent.current_state in agent.state_map_action_map_qobj:
            action_map_qobj = agent.state_map_action_map_qobj[agent.current_state]
            current_q_obj = action_map_qobj[agent.last_action]
        else:
            current_q_obj = QObject(q_value=0, n=0, e=0)

        prev_action_map_qobj = agent.state_map_action_map_qobj[agent.prev_state]
        prev_q_obj = prev_action_map_qobj[agent.second_last_action]

        # deltaの更新
        delta = agent.last_reward + self.gamma * current_q_obj.q_value - prev_q_obj.q_value
        # e(s, a)の更新
        prev_q_obj.e += 1
        # nの更新
        prev_q_obj.n += 1

        # すべてのs, aに対して。
        for state, action_map_qobj in agent.state_map_action_map_qobj.items():
            for action, qobj in action_map_qobj.items():
                # Q値の更新
                qobj.q_value += self.alpha * delta * qobj.e
                # e(s, a)の更新
                qobj.e *= self.gamma * self.lamb

