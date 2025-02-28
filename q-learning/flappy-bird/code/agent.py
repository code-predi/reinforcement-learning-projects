import flappy_bird_gymnasium
import gymnasium 
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hyperparam_file = r'E:\RL-proj\flappy-bird\rl-flappy-bird\code\hyperparameters.yml'
class Agent:
    def __init__(self, hyperparameter_set):
        with open(hyperparam_file) as file: 
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]
            #print(hyperparameters)
        
        self.replay_memory_size = hyperparameters['replay_memory_size']   # size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size']         # size of the training data set sampled from memory
        self.epsilon_init = hyperparameters['epsilon_init']               # 1 = 100% random actions
        self.epsilon_decay = hyperparameters['epsilon_decay']             # epsilon decay rate
        self.epsilon_min = hyperparameters['epsilon_min']                 # minimum epsilon value

    def run(self, render=False, is_trainning = True):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        epsilon = self.epsilon_init

        #to keep track of rewards earned in each episode
        rewards_per_episode = []

        #to keep track of epsilon decay after each episode
        epsilon_history = []

        #initializing DQN model
        policy_dqn = DQN(state_dim=num_states, action_dim=num_actions).to(device)

        if is_trainning: 
            memory = ReplayMemory(maxlen=self.replay_memory_size)

        for episode in itertools.count():
            state, _ = env.reset()
            #Converting state to tensors datatype, since pytorch only accepts tensors as inputs
            state = torch.tensor(data=state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0
            #Since we are running the trainning indefinetly
            while not terminated: 
            
                #Next Action
                # (feed the observation to your agent here)
                if is_trainning and random.random() < epsilon:
                    action = env.action_space.sample()
                    #Converting action to tensors datatype.
                    action = torch.tensor(data=action, dtype=torch.int64, device=device)
                else: 
                    #Here we will add an extra dimension 👇,  since pytorch takes the first value in shape as the batch number. basicall go from tensor([1,2,3]) to tensor([[1,2,3]])
                    #Now later on we will remove the extra dimentsion👇 to perform argmax() 
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing
                new_state, reward, terminated, _, info = env.step(action=action.item())
                
                #Converting new_state and reward to tensors datatype.
                new_state = torch.tensor(data=new_state, device=device, dtype=torch.float)
                reward = torch.tensor(data=reward, dtype=torch.float, device=device)

                if is_trainning: 
                    memory.append((state, action, new_state, reward, terminated))
                
                #Accumulate reward 
                episode_reward += reward

                #Move to new state
                state = new_state

                # Check if the player is still live
                #Since we are running the trainning indefinetly, we will comment the terminate and env.close() statement
                #if terminated:
                #    break
            #env.close()
            
            #storing rewards achieved per episode in the rewards_per_episdoe list 
            rewards_per_episode.append(episode_reward)
            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)


if __name__ == '__main__':
    ag = Agent('cartpole1')
    ag.run(is_trainning=True, render=True)