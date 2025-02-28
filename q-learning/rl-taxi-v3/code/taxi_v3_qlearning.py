import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle as pkl


def run(episodes, is_training=True, is_render=False):
    env = gym.make('Taxi-v3', render_mode='human' if is_render else None)

    if is_training: 
        q = np.zeros((env.observation_space.n, env.action_space.n)) #500x6 array
    else: 
        f = open('taxi.pkl', 'rb')
        q = pkl.load(f)
        f.close()

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1 
    decay_rate =0.0001
    rng = np.random.default_rng(seed=42)

    rewards_per_episode = np.zeros(episodes)

    # Start timing the training
    start_time = time.time()

    for i in range(episodes): 
        state = env.reset()[0]
        terminated = False 
        truncated = False

        rewards = 0
        print(f"{i} epsisodes out {episodes} done")
        while(not terminated and not truncated): 
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else: 
                action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action=action)

            rewards += reward

            if is_training: 
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )
        
            state = new_state
        epsilon = max(epsilon - decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001
        
        rewards_per_episode[i] = rewards
    
    env.close()
    # End timing the training
    training_time = time.time() - start_time

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')
    if is_training: 
        f = open('taxi.pkl', 'wb')
        pkl.dump(q,f)
        f.close()
        print(training_time)

if __name__ == "__main__": 
    run(15000)

    run(10, is_training=False, is_render=True)