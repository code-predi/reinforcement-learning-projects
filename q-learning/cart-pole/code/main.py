import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl 

def run(is_training=False, is_render=False): 

    env = gym.make("CartPole-v1", render_mode='human' if is_render else None)

    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4,4,10)

    if is_training:
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else: 
        f = open('cartpole.pkl', 'rb')
        q = pkl.load(f)
        f.close()
    
    learning_rate_a = 0.1
    discount_rate_g = 0.99

    epsilon = 1
    epsilon_decay_rate = 0.00001

    rng = np.random.default_rng()

    rewards_per_episode = []
    i = 0

    while(True): 

        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False 

        rewards=0

        while(not terminated and rewards < 10000):

            if (is_training and rng.random() < epsilon):
                action = env.action_space.sample()
            else: 
                action = np.argmax(q[state_p, state_v, state_a, state_av, :]) 

            new_state, reward, terminated, _, _ = env.step(action=action)

            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)     
            new_state_a = np.digitize(new_state[2], ang_space)     
            new_state_av = np.digitize(new_state[3], ang_vel_space)        

            if (is_training): 
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_rate_g * (np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q[state_p, state_v, state_a, state_av, action])
                )  

            state = new_state
            state_p = new_state_p
            state_v = new_state_v 
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

            if not is_training and rewards%100==0: 
                print(f"Episode: {i} Rewards: {rewards}")

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100 :])

        if is_training and i%100==0: 
            print(f"Episode: {i} {rewards} Epsilon: {epsilon:0.2f} Mean Rewards {mean_rewards:0.1f}")

        if mean_rewards>1000: 
            break
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        i+=1
    
    env.close()

    if is_training: 
        f = open('cartpole.pkl', 'wb')
        pkl.dump(q,f)
        f.close()

    mean_rewards = []
    for t in range(i): 
        mean_rewards.append(np.mean(rewards_per_episode[max(t-100, 0):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole.png')

if __name__ == "__main__": 
    #run(is_training=True)
    run(is_render=True)
