from DeepQNetwork import DQN
from TetrisBoard import TetrisBoard
from Agent import Agent

import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import torch


#  from https://github.com/philtabor/Youtube-Code-Repository
def plot_learning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

if __name__ == "__main__":
    env = TetrisBoard()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=10 * 4, input_dims=[7 + 10], lr=0.0003)
    scores, eps_history = [], []
    n_games = 15000
    best_score = -99999
    score = -99999
    best_agent = None
    n_iterations = 0

    for i in range(n_games):

        if score > best_score:
            best_score = score
            best_agent = agent

        score = 0
        done = False
        env.reset()
        observation = env.get_game_state()
        while not done:
            action = agent.choose_action(observation[0])
            observation_, reward, done, info = env.step(action, observation[1])
            score += reward
            agent.store_transition(observation[0], action, reward, observation_[0], done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        print(f"game: {i}, score: {score}, epsilone: {agent.epsilon}, cleared lines: {env.get_cleared_lines()}")
    x = [i + 1 for i in range(n_games)]
    filename = "tenth_iteration/learning_history.png"
    plot_learning(x, scores, eps_history, filename)

    best_agent.save_model("tenth_iteration/model")
    #
    # env = TetrisBoard(False)
    # model = DQN(n_actions=10 * 4, input_dims=[7 + 10], lr=0.0003, fc1_dims=256, fc2_dims=256, fc3_dims=128)
    # model.load_state_dict(torch.load("tenth_iteration/model"))
    # model.eval()
    # observation = env.get_game_state()
    #
    # while True:
    #     state = torch.tensor(np.array(observation[0])).to(model.device)
    #     actions = model.forward(state)
    #     action = torch.argmax(actions).item()
    #     data = action / 10
    #     rotation = int(data)
    #     column = round(((data - int(data)) * 10))
    #     env.place_piece(observation[1], column, rotation)
    #     observation = env.get_game_state()
    #     if env.is_dead():
    #         print("death")
    #         break
