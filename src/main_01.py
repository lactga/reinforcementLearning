from src.main_class.MainClassOnlineQLearning import MainClassOnlineQLearning
from src.environments.TreasureHuntingEnvironment import TreasureHuntingEnvironment
from src.agents.AgentQTable import AgentQtable
from src.explorers.EpsilonGreedy import EpsilonGreedy
from src.learners.QLearning import QLearning

if __name__ == '__main__':

    # 環境
    environment = TreasureHuntingEnvironment(p=1.0, reward_value=10)

    # エージェント
    explorer = EpsilonGreedy(epsilon=0.8)
    learner = QLearning(alpha=0.1, gamma=0.9)
    agent = AgentQtable(explorer=explorer, learner=learner)

    # メイン処理
    main_class = MainClassOnlineQLearning(environment=environment, agent=agent, main_loop_num=1000)
    main_class.run_main_loop()
