from controller import Controller

MIN_GREEN_VEHICLE = 20
MAX_RED_VEHICLE = 30

class SOTL(Controller):
    def make_action(self, state):
        currentLogic, numVehOrdered = state
        numberVehOnGreenLane = 0
        numberVehOnRedLane = 0

        for i in range(len(numVehOrdered)):
            if currentLogic[i] in ['r', 'R']:
                numberVehOnRedLane += numVehOrdered[i]
            elif currentLogic[i] in ['g', 'G']:
                numberVehOnGreenLane += numVehOrdered[i]
            else:
                print(state, "Error")
        if (numberVehOnGreenLane < MIN_GREEN_VEHICLE and numberVehOnRedLane > MAX_RED_VEHICLE) or (numberVehOnGreenLane == 0 and numberVehOnRedLane > 0):
            return 1
        return 0