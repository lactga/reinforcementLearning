import matplotlib.pyplot as plt
import pickle


def main():
    with open('out/greedy.dmp', 'rb') as f:
        greedy = pickle.load(f)
    with open('out/epsilon_greedy_0.1.dmp', 'rb') as f:
        epsilon_greedy_01 = pickle.load(f)
    with open('out/epsilon_greedy_0.01.dmp', 'rb') as f:
        epsilon_greedy_001 = pickle.load(f)
    with open('out/softmax_1.dmp', 'rb') as f:
        softmax_1 = pickle.load(f)
    with open('out/softmax_0.1.dmp', 'rb') as f:
        softmax_01 = pickle.load(f)
    with open('out/softmax_0.01.dmp', 'rb') as f:
        softmax_001 = pickle.load(f)

    x = tuple(range(1, 1000 + 1))
    plt.plot(x, greedy, label='greedy')
    plt.plot(x, epsilon_greedy_01, label='epsilon-greedy 0.1')
    plt.plot(x, epsilon_greedy_001, label='epsilon-greedy 0.01')
    plt.plot(x, softmax_1, label='softmax-1')
    plt.plot(x, softmax_01, label='softmax-0.1')
    plt.plot(x, softmax_001, label='softmax-0.01')
    plt.legend(loc='lower right')  # 凡例を表示
    plt.show()

if __name__ == '__main__':
    main()
