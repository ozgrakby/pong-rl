import config
import numpy as np
import random
from collections import deque, namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=config.MEMORY_SIZE)

    def push(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self):
        experiences = random.sample(self.buffer, config.BATCH_SIZE)

        batch = Experience(*zip(*experiences))

        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action, dtype=np.int64)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)

        dones = np.array(batch.done, dtype=np.float32)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)