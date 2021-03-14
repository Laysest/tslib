import traci
from SOTL import SOTL
class TrafficLight:
    def __init__(self, id, algorithm='SOTL', yellowDuration=3):
        self.id = id
        self.controlAlgorithm = algorithm
        self.yellowDuration = yellowDuration
        self.controlAlgorithm = algorithm
        self.lanes = traci.trafficlight.getControlledLanes(self.id) # check
        
        # Create controller based on the algorithm configed
        if self.controlAlgorithm == 'SOTL':
            self.controller = SOTL()

    def get_state(self):
        """
            return the current state of the intersection, must depend on the control algorithm
        """
        if self.controlAlgorithm == 'SOTL':
            allLogic = traci.trafficlight.getAllProgramLogics(self.id)[0]
            currentLogic = allLogic.getPhases()[allLogic.currentPhaseIndex].state
            numVehOrdered = []
            for lane in self.lanes:
                numVehOrdered.append(traci.lane.getLastStepVehicleNumber(lane))
            
            return currentLogic, numVehOrdered

    def update(self):
        self.controller.make_action(self.get_state)