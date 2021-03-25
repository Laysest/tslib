from SOTL import SOTL
from SimpleRL import SimpleRL
import random

MAX_INT = 9999999
EXPLORE_PROBABILITY = 0.05

class TrafficLight:
    def __init__(self, tfID, algorithm='SimpleRL', yellow_duration=3, traci=None, cycle_control=5):
        self.id = tfID
        self.control_algorithm = algorithm
        self.yellow_duration = yellow_duration
        self.control_algorithm = algorithm
        self.traci = traci
        self.cycle_control = 5
        # traci.setOrder(2)

        self.lanes = self.traci.trafficlight.getControlledLanes(tfID) # check
        # self.lanes = []
        # Create controller based on the algorithm configed
        if self.control_algorithm == 'SOTL':
            self.controller = SOTL()
        elif self.control_algorithm == 'SimpleRL':
            self.controller = SimpleRL()
        else:
            print("Must implement method named %s" % algorithm)

        self.current_phase = self.traci.trafficlight.getPhase(tfID)
        self.control_actions = []
        self.setLogic()
        self.last_action, self.last_state = None, None

    def setLogic(self):
        """
            ### To do ###
            Set logic program for the traffic light
            Restart the logic at phase 0
        """
        self.traci.trafficlight.setPhase(self.id, 0)
        self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

    def getState(self):
        """
            return the current state of the intersection, must depend on the control algorithm
        """
        if self.control_algorithm == 'SOTL':
            all_logic_ = self.traci.trafficlight.getAllProgramLogics(self.id)[0]
            current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
            num_veh_ordered = []
            for lane in self.lanes:
                num_veh_ordered.append(self.traci.lane.getLastStepVehicleNumber(lane))
            
            return current_logic, num_veh_ordered

        elif self.control_algorithm == 'SimpleRL':
            all_logic_ = self.traci.trafficlight.getAllProgramLogics(self.id)[0]            
            current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
            num_veh_ordered = []
            for lane in self.lanes:
                num_veh_ordered.append(self.traci.lane.getLastStepVehicleNumber(lane))

            number_veh_on_green_lanes = 0
            number_veh_on_red_lanes = 0
            for i in range(len(num_veh_ordered)):
                if current_logic[i] in ['r', 'R']:
                    number_veh_on_red_lanes += num_veh_ordered[i]
                elif current_logic[i] in ['g', 'G']:
                    number_veh_on_green_lanes += num_veh_ordered[i]
                else:
                    print("Error in getState in case of SimpleRL")
                    print("step: %d, id: %s, current_logic: %s, actions: %s" % (self.traci.simulation.getTime(), self.id, str(current_logic), str(self.control_actions)))

            return [number_veh_on_green_lanes, number_veh_on_red_lanes]

    def computeReward(self):
        if self.control_algorithm == 'SimpleRL':
            reward = 0
            for lane in self.lanes:
                reward -= self.traci.lane.getLastStepHaltingNumber(lane)
            return reward

    def update(self, is_train=False):
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
    #                   check if has just deleted yellow phase (still have action in action_queue) => change phase

        # all_logic_ = self.traci.trafficlight.getAllProgramLogics(self.id)[0]            
        # current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
        # print("step: %d, id: %s, control_actions: %s, current_logic: %s" % (self.traci.simulation.getTime(), self.id, self.control_actions, current_logic))

        if len(self.control_actions) <= 0: 
            cur_state = self.getState()
            # if is training:
            #     check to explore
            if is_train:
                if random.uniform(0, 1) <= EXPLORE_PROBABILITY:
                    action = random.randint(0, 1)
                else:
                    action = self.controller.makeAction(cur_state)
                # log last_state, last_action, reward, cur_state
                if self.last_state != None and self.last_action != None:
                    # compute reward
                    reward = self.computeReward()
                    self.controller.exp_memory.add([self.last_state, self.last_action, reward, cur_state])
                self.last_state, self.last_action = cur_state, action
            else:
                action = self.controller.makeAction(cur_state)

            if action == 1:
                current_phase_ = self.traci.trafficlight.getPhase(self.id)
                if current_phase_ < 3:
                    self.traci.trafficlight.setPhase(self.id, current_phase_ + 1)
                    self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)
                else:
                    print("******* error := change at yellow phase ??? ")

                self.control_actions.extend([{'type': 'yellow_phase', 'length': self.yellow_duration},
                                            {'type': 'red_green_phase', 'length': self.cycle_control}])
            else:   
                self.control_actions.append({'type': 'red_green_phase', 'length': self.cycle_control})
        
        if len(self.control_actions) > 0:
            self.control_actions[0]['length'] -= 1
            if self.control_actions[0]['length'] <= 0:
                self.control_actions.pop(0)
                if len(self.control_actions) > 0:
                    current_phase_ = self.traci.trafficlight.getPhase(self.id)
                    if current_phase_ == 1:
                        self.traci.trafficlight.setPhase(self.id, current_phase_ + 1)
                    elif current_phase_ == 3:
                        self.traci.trafficlight.setPhase(self.id, 0)
                    else:
                        print("******* error in control: current_phase: %d, control_actions: %s" % (current_phase_, self.control_actions))
                    self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

        self.current_phase = self.traci.trafficlight.getPhase(self.id)

    def replay(self):
        self.controller.replay()