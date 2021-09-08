from src.controller.controller import Controller, ActionType
from collections import deque
import numpy as np
import random
from sklearn.model_selection import train_test_split
from src.glo_vars import GloVars
import sys

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, BATCH_SIZE):
        return random.sample(self.buffer, GloVars.BATCH_SIZE)
    
    def len(self):
        return len(self.buffer)

class RLAgent(Controller):
    def __init__(self, cycle_control, input_space, output_space):
        Controller.__init__(self)
        self.cycle_control = cycle_control
        self.output_space = output_space
        self.model = self.buildModel(input_space, output_space)
        self.exp_memory = Memory(GloVars.MEMORY_SIZE)

    def buildModel(self, input_space, output_space):
        print("Must <<overwrite>> \"build_mode\" function in RL-based methods")
        sys.exit(0)

    def processState(self, state):
        print("Must <<override>> processState(state)!!")
        pass

    def computeReward(self, state, historical_data):
        print("Muss <<override>> computeReward(state)")
        pass
        
    def replay(self):
        if self.exp_memory.len() < GloVars.SAMPLE_SIZE:
            return
        minibatch =  self.exp_memory.sample(GloVars.SAMPLE_SIZE)    
        batch_states = []
        batch_targets = []
        for state_, action_, reward_, next_state_ in minibatch:
            qs = self.model.predict(np.array([next_state_]))
            target = reward_ + GloVars.GAMMA*np.amax(qs[0])
            target_f = self.model.predict(np.array([state_]))
            target_f[0][action_] = target
            batch_states.append(state_)
            batch_targets.append(target_f[0])
        
        self.model.fit(np.array(batch_states), np.array(batch_targets), epochs=GloVars.EPOCHS, batch_size=GloVars.BATCH_SIZE, shuffle=False, verbose=0, validation_split=0.3)

    def makeAction(self, state):
        state_ = self.processState(state)
        out_ = self.model.predict(np.array([state_]))[0]
        action = np.argmax(out_)
        if 2*action == state['current_phase_index']:
            return action, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]
        return action, [{'type': ActionType.CHANGE_TO_PHASE, 'phase_index': action*2, 'length': self.cycle_control, 'executed': False}]

    def randomAction(self, state):
        action = random.randint(0, self.output_space - 1)
        if 2*action == state['current_phase_index']:
            return action, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]
        return action, [{'type': ActionType.CHANGE_TO_PHASE, 'phase_index': action*2, 'length': self.cycle_control, 'executed': False}] 