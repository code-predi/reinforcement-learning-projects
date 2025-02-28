import random 
from collections import deque

#This deque will store: 
# state(current), action, new_state, reward, terminated

class ReplayMemory(): 
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None: 
            random.seed(seed)

    def append(self, transition): 
        self.memory.append(transition)

    def sampe(self, sample_size): 
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    