import numpy as np
import matplotlib.pyplot as plt
from src.explorers.EpsilonGreedy import EpsilonGreedy

def plot_maze(main_class):
    epsilon_greedy = EpsilonGreedy(epsilon=0)

    X = []
    Y = []
    U = []
    V = []
    R = []
    for state, action_map_qobj in main_class.agent.state_map_action_map_qobj.items():
        action = epsilon_greedy.select_action(action_map_qobj)
        tmpy, tmpx = state
        tmpv, tmpu = action
        tmpr = max([qobj.q_value for qobj in action_map_qobj.values()])
        X.append(tmpx)
        Y.append(tmpy)
        U.append(tmpu)
        V.append(-tmpv)
        R.append(tmpr)

    fig, ax = plt.subplots()

    image = 9 - main_class.environment.maze.maze_list
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('Maze')

    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.quiver(X, Y, U, V, R, alpha=.5)
    plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)

    plt.show()
