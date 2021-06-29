from enum import Enum

class ActionType(Enum):
    CHOICE_OF_PHASE = 1
    CHANGING_KEEPING = 2
    DEFINE_NEXT_CYCLE = 3
    CHANGE_TO_NEXT_PHASE = 4
    KEEP_PHASE = 5
    YELLOW_PHASE = 6
    CHANGE_TO_PHASE = 7

# class ControlAlgorithm(Enum):
#     FixedTime = 1 # TODO
#     SOTL = 2
#     MaxPressure = 3 # TODO
#     CDRL = 4
#     VFB = 5
#     IntelliLight = 6
class Controller:
    def __init__(self):
        pass

    def makeAction(self, state):
        print("Must <<override> makeAction(state)!!")
        return 0, [0]

    def processState(self, state):
        print("Must <<override> processState(state)!!")
        pass