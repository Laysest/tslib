from enum import Enum

class ActionType(Enum):
    CHOICE_OF_PHASE = 1
    CHANGING_KEEPING = 2

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
        pass

    def processState(self, state):
        print("Must <<override> processState(state)!!")
        pass

    def actionType(self):
        return ActionType.CHOICE_OF_PHASE