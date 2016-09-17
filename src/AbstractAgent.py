from abc import ABCMeta, abstractmethod


class AbstractAgent(metaclass=ABCMeta):
    """
    強化学習のエージェントの抽象クラス
    """

    @abstractmethod
    def observe(self):
        """
        環境、報酬を観測する
        :return:
        """
        pass

    @abstractmethod
    def train(self):
        """
        方策、価値観数などの学習(更新)を行う
        :return:
        """
        pass

    @abstractmethod
    def select_action(self):
        """
        可能な行動から行動を選択する
        :return:
        """
        pass


