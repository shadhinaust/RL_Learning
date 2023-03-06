import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('CartPole-v0')
state_space = 4
action_space = 2

def Q_table(state_space, action_space, bin_size=30):
    bins = [np.linspace(-4.8, 4.8, bin_size),
            np.linspace(-4, 4, bin_size),
            np.linspace(-0.418, 0.418, bin_size),
            np.linspace(-4, 4, bin_size)]

    q_table = np.random.uniform(low=1, high=1, size=(
        [bin_size]*state_space + [action_space]))
    return q_table, bins

def discrete(state, bins):
    index = []
    for i in range(len(state)):
        index.append(np.digitize(state[i], bins[i])-1)
    return tuple(index)

def Q_learning(q_table, bins, episodes=5000, gamma=0.95, lr=0.1, timestep=100, epsilon=0.2):
    rewards = 0
    solved = False
    steps = 0
    runs = [0]
    data = {'max': [0], 'avg': [0]}
    ep = [i for i in range(0, episodes+1, timestep)]

    for episode in range(1, episodes+1):
        print('Episode: {}'.format(episode))
        i_state = np.array(env.reset()[0])
        current_state = discrete(i_state, bins)
        score = 0
        terminated = False

        while not terminated:
            steps += 1
            print('Step: {}'.format(steps))
            ep_start = time.time()
            if episode % timestep == 0:
                env.render()

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state])

            observation, reward, terminated, truncated, info = env.step(action)
            next_state = discrete(observation, bins)
            score += reward

            if not terminated:
                max_future_q = np.max(q_table[next_state])
                current_q = q_table[current_state+(action,)]
                new_q = (1-lr)*current_q + lr*(reward + gamma*max_future_q)
                q_table[current_state, (action,)] = new_q

            current_state = next_state
        else:
            rewards += score
            runs.append(score)
            if score > 195 and steps >= 100 and solved == False:
                solved = True
                print('Solved in episode: {} in time {}'.format(
                    episode, (time.time()-ep_start)))

        if episode % timestep == 0:
            print('Episode: {} | Reward -> {} | Max Reward: {} |Time: {}'.format(episode,
                  rewards/timestep, max(runs), time.time() - ep_start))
            data['max'].append(max(runs))
            data['avg'].append(rewards/timestep)
            if rewards / timestep >= 195:
                print('Solved in episode: {}'.format(episode))
            rewards, runs = 0, [0]

    if len(ep) == len(data['max']):
        plt.plot(ep, data['max'], label='Max')
        plt.plot(ep, data['avg'], label='Avg')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")

    env.close()

q_table, bins = Q_table(len(env.observation_space.low), env.action_space.n)
Q_learning(q_table=q_table, bins=bins, lr=0.5, gamma = 0.5, episodes=5*10**3, timestep=1000)