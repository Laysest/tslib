from controller import Controller
from collections import deque

MEMORY_SIZE = 2048

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
        self.model = None
        self.experience_memory = Memory(MEMORY_SIZE)
    
    def build_model(self):
        pass

    def replay(self):
        pass

    def make_action(self, state):
        return self.model.predict(state)