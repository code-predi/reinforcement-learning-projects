import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as pkl 

def run(episodes, is_training= True, is_render=False): 
    env = gym.make("MountainCar-v0", render_mode='human' if is_render else None)

    #Dividing position and velocity into multiple segments 

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)  #Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)  #Between -0.07 and 0.07

    if(is_training): 
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else: 
        f = open('mountaincarq.pkl', 'rb')
        q = pkl.load(f)
        f.close()
    
    learning_rate_a = 0.5
    discount_rate_g = 0.3
    epsilon = 1 
    epsilon_decay_rate = 2/episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes): 
        state = env.reset()[0]
        terminated = False 
        rewards = 0 
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        print(f"{i} episode out of {episodes} Completed")
        while(not terminated and rewards>-1000):
            if is_training and rng.random()<epsilon:
                action = env.action_space.sample()
            else: 
                action = np.argmax(q[state_p, state_v, :])
        
            new_state, reward, terminated, _, _ = env.step(action=action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training: 
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_rate_g * (np.max(q[new_state_p,new_state_v, :]) - q[state_p, state_v, action]) 
                )
            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon-epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    env.close()

    if is_training: 
        f = open('mountaincarq.pkl','wb')
        pkl.dump(q,f)
        f.close()
    
    mean_rewards = np.zeros(episodes)
    for t in range(episodes): 
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig('mountaincar.png')

if __name__ == "__main__": 
    #run(5000, is_render=False)

    run(10, is_training=False, is_render=True)