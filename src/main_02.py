from src.main_class.MainClassOnlineSarsa import MainClassOnlineSarsa
from src.environments.TreasureHuntingEnvironment import TreasureHuntingEnvironment
from src.agents.AgentQTable import AgentQtable
from src.explorers.EpsilonGreedy import EpsilonGreedy
from src.learners.Sarsa import Sarsa

if __name__ == '__main__':

    # 環境
    # environment = TreasureHuntingEnvironment(p=1.0, reward_value=10)
    environment = TreasureHuntingEnvironment(p=0.8, reward_value=10)
    # environment = TreasureHuntingEnvironment(p=0.6, reward_value=10)
    # environment = TreasureHuntingEnvironment(p=0.51, reward_value=10)

    # エージェント
    explorer = EpsilonGreedy(epsilon=0.1)
    learner = Sarsa(alpha=0.1, gamma=0.9)
    agent = AgentQtable(explorer=explorer, learner=learner)

    # メイン処理
    main_class = MainClassOnlineSarsa(environment=environment, agent=agent, episode_loop_num=1000, main_loop_num=1000)
    main_class.run_main_loop()
