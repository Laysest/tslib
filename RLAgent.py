from controller import Controller
from collections import deque
import numpy as np

MEMORY_SIZE = 2048
BATCH_SIZE = 64
GAMMA = 0.95
EPOCHS = 50
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def len(self):
        return len(self.buffer)

class RLAgent(Controller):
    def __init__(self):
        self.model = self.build_model()
        self.experienceMemory = Memory(MEMORY_SIZE)

    def build_model(self):
        print("Must <<overwrite>> \"build_mode\" function in RL-based methods")
        return None

    def replay(self):
        if self.experienceMemory.len() < BATCH_SIZE:
            return
        minibatch =  self.experienceMemory.sample(BATCH_SIZE)    
        batch_states = []
        batch_targets = []
        for state_, action_, reward_, next_state_ in minibatch:
            qs = self.model.predict(np.array([next_state_]))
            target = reward_ + GAMMA*np.amax(qs[0])
            target_f = self.model.predict(np.array([state_]))
            target_f[0][action_] = target

            batch_states.append(state_)
            batch_targets.append(target_f[0])
        
        self.model.fit(np.array(batch_states), np.array(batch_targets), epochs=EPOCHS, shuffle=False, verbose=0, validation_split=0.3)

    def make_action(self, state):
        return np.argmax(self.model.predict(np.array([state]))[0])