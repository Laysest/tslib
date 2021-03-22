from SOTL import SOTL
from SimpleRL import SimpleRL
import random

MAX_INT = 9999999
EXPLORE_PROBABILITY = 0.05

class TrafficLight:
    def __init__(self, tfID, algorithm='SimpleRL', yellowDuration=3, traci=None, cycleControl=5):
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
        elif self.controlAlgorithm == 'SimpleRL':
            self.controller = SimpleRL()
        else:
            print("Must implement method named %s" % algorithm)

        self.currentPhase = self.traci.trafficlight.getPhase(tfID)
        self.controlActions = []
        self.set_logic()
        self.lastAction, self.lastState = None, None

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

        elif self.controlAlgorithm == 'SimpleRL':
            allLogic = self.traci.trafficlight.getAllProgramLogics(self.id)[0]
            currentLogic = allLogic.getPhases()[allLogic.currentPhaseIndex].state
            numVehOrdered = []
            for lane in self.lanes:
                numVehOrdered.append(self.traci.lane.getLastStepVehicleNumber(lane))

            numberVehOnGreenLane = 0
            numberVehOnRedLane = 0
            for i in range(len(numVehOrdered)):
                if currentLogic[i] in ['r', 'R']:
                    numberVehOnRedLane += numVehOrdered[i]
                elif currentLogic[i] in ['g', 'G']:
                    numberVehOnGreenLane += numVehOrdered[i]
                else:
                    print("Error in get_state of SimpleRL")
                    print(currentLogic)

            return [numberVehOnGreenLane, numberVehOnRedLane]

    def compute_reward(self):
        if self.controlAlgorithm == 'SimpleRL':
            reward = 0
            for lane in self.lanes:
                reward -= self.traci.lane.getLastStepHaltingNumber(lane)
            return reward

    def update(self, isTrain=False):
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
            curState = self.get_state()
            # if is training:
            #     check to explore
            if isTrain:
                if random.uniform(0, 1) <= EXPLORE_PROBABILITY:
                    action = random.randint(0, 1)
                else:
                    action = self.controller.make_action(curState)
                # log lastState, lastAction, reward, curState
                if self.lastState != None and self.lastAction != None:
                    # compute reward
                    reward = self.compute_reward()
                    self.controller.experienceMemory.add([self.lastState, self.lastAction, reward, curState])
                self.lastState, self.lastAction = curState, action
            else:
                action = self.controller.make_action(curState)

            if action == 1:
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

    def replay(self):
        self.controller.replay()