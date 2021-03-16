from SOTL import SOTL

MAX_INT = 9999999

class TrafficLight:
    def __init__(self, tfID, algorithm='SOTL', yellowDuration=3, traci=None, cycleControl=5):
        self.id = tfID
        self.controlAlgorithm = algorithm
        self.yellowDuration = yellowDuration
        self.controlAlgorithm = algorithm
        self.traci = traci
        self.cycleControl = 5
        # traci.setOrder(2)

        self.lanes = self.traci.trafficlight.getControlledLanes(tfID) # check
        # self.lanes = []
        # Create controller based on the algorithm configed
        if self.controlAlgorithm == 'SOTL':
            self.controller = SOTL()

        self.currentPhase = self.traci.trafficlight.getPhase(tfID)
        self.controlActions = []
        self.set_logic()
    
    def set_logic(self):
        """
            ### To do ###
            Set logic program for the traffic light
            Restart the logic at phase 0
        """

        self.traci.trafficlight.setPhase(self.id, 0)
        self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

    def get_state(self):
        """
            return the current state of the intersection, must depend on the control algorithm
        """
        if self.controlAlgorithm == 'SOTL':
            allLogic = self.traci.trafficlight.getAllProgramLogics(self.id)[0]
            currentLogic = allLogic.getPhases()[allLogic.currentPhaseIndex].state
            numVehOrdered = []
            for lane in self.lanes:
                numVehOrdered.append(self.traci.lane.getLastStepVehicleNumber(lane))
            
            return currentLogic, numVehOrdered

    def update(self):
    #   if len = 0 => no action in queue:
    #       get action by state
    #       if action = True:
    #           change & add to queue including yellow and next cycle
    #       else:
    #           add to queue next cycle
    #   if len >= 0 => have action in queue:
    #       -1 time length
    #           if time length == 0:
    #               that means finish the cycle => delete in queue
    #                   check if has just deleted yellow phase => change phase
        if len(self.controlActions) <= 0:
            action = self.controller.make_action(self.get_state())
            if action == True:
                if self.currentPhase < 3:
                    self.traci.trafficlight.setPhase(self.id, self.currentPhase + 1)
                else:
                    self.traci.trafficlight.setPhase(self.id, 0)
                self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

                self.controlActions.extend([{'type': 'yellow_phase', 'length': self.yellowDuration},
                                            {'type': 'red_green_phase', 'length': self.cycleControl}])
            else:
                self.controlActions.append({'type': 'red_green_phase', 'length': self.cycleControl})
        
        if len(self.controlActions) > 0:
            self.controlActions[0]['length'] -= 1
            if self.controlActions[0]['length'] <= 0:
                self.controlActions.pop(0)
                if len(self.controlActions) > 0:
                    if self.currentPhase < 3:
                        self.traci.trafficlight.setPhase(self.id, self.currentPhase + 1)
                    else:
                        self.traci.trafficlight.setPhase(self.id, 0)
                    self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

        self.currentPhase = self.traci.trafficlight.getPhase(self.id)