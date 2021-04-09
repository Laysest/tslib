from controller import Controller
from collections import deque
import numpy as np
import random

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
        self.model = self.buildModel()
        self.exp_memory = Memory(MEMORY_SIZE)

    def buildModel(self):
        print("Must <<overwrite>> \"build_mode\" function in RL-based methods")
        return None

    def processState(self, state):
        print("Must <<override> processState(state)!!")
        pass

    def computeReward(self, state):
        print("Muss <<override>> computeReward(state)")
        pass
    
    def replay(self):
        if self.exp_memory.len() < BATCH_SIZE:
            return
        minibatch =  self.exp_memory.sample(BATCH_SIZE)    
        batch_states = []
        batch_targets = []
        for state_, action_, reward_, next_state_ in minibatch:
            qs = self.model.predict([next_state_])
            target = reward_ + GAMMA*np.amax(qs[0])
            target_f = self.model.predict([state_])
            target_f[0][action_] = target

            batch_states.append(state_)
            batch_targets.append(target_f[0])
        
        self.model.fit(batch_states, batch_targets, epochs=EPOCHS, shuffle=False, verbose=0, validation_split=0.3)

    def makeAction(self, state):
        """
            Return the action and a boolean value indicating the action is a choice of phases or changing/keeping.
            For example, IntelliLight is a changing/keeping action, so this function will be overwriten
        """

        state_ = self.processState(state)
        out_ = self.model.predict([state_])[0]

        return np.argmax(out_)