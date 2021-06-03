"""
    This file declares a class for global variables
"""
import traci

class GloVars:
    """
        This class contain global variables and no function.
    """
    ACTION_SPACE = 2
    EXPLORE_PROBABILITY = 0.05
    ARRAY_LENGTH = 9
    CENTER_LENGTH = 1
    MAP_SIZE = (2*(ARRAY_LENGTH + CENTER_LENGTH), 2*(ARRAY_LENGTH + CENTER_LENGTH))
    STATE_SPACE = (MAP_SIZE[0], MAP_SIZE[1], 1)
    STATE_SPACE_TWO_CHANNELS = (MAP_SIZE[0], MAP_SIZE[1], 2)
    LENGTH_CELL = 5
    BATCH_SIZE = 64
    GAMMA = 0.95
    EPOCHS = 50
    SAMPLE_SIZE = 256 
    INTERVAL = 300
    MEMORY_SIZE = 2048
    step = 0
    traci = traci
    config = None
