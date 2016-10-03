from src.agents.AbstractAgent import AbstractAgent


class Agent(AbstractAgent):
    """
    多腕バンディットアルゴリズムのエージェントクラス
    """
    def __init__(self, estimate_q_value_cls, policy_cls, action_map_qobj):
        """
        インストラクタ
        """
        self.action_map_qobj = action_map_qobj
        self.estimate_q_value_cls = estimate_q_value_cls
        self.policy_cls = policy_cls
        self.recent_action = None

    def observe(self, reward):
        """
        報酬を観測する
        :param reward: 報酬
        :return:
        """
        self.train(reward)

    def train(self, reward):
        """
        学習を行う
        :return:
        """
        q_obj = self.action_map_qobj[self.recent_action]
        self.estimate_q_value_cls.estimate_q_value(reward, q_obj)

    def select_action(self):
        """
        可能な行動から行動を選択する
        :return:
        """
        action = self.policy_cls.select_action(self.action_map_qobj)
        self.recent_action = action
        return action
