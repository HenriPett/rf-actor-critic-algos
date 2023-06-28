import gymnasium as gym
from control_algo import Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    agent = Agent(eps=0.001)
    n_episodes = 200000
    win_lose_draw = {-1: 0, 0: 0, 1: 0}
    win_rate = []

    for i in range(n_episodes):
        if i > 0 and 1000 == 0:
            pct = win_lose_draw[1] / i
            win_rate.append(pct)
            print(f"Starting episode {i}", end="", flush=True)
        if i % 50000 == 0:
            rates = win_rate[-1] if win_rate else 0.0
            print(f"Starting episode {i}, win rate {rates}")

        observation = env.reset()[0]
        done = False
        truncated = False

        while not done or not truncated:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            agent.memory.append((observation, action, reward))
            observation = observation_
        agent.update_Q()
        win_lose_draw[reward] += 1
    plt.plot(win_rate)
    plt.show()
