from SOTL import SOTL
from SimpleRL import SimpleRL
from CDRL import CDRL
from VFB import VFB
from IntelliLight import IntelliLight
import random
import sys
import sumolib
from controller import ActionType
from GlobalVariables import GlobalVariables
import tensorflow as tf

MAX_INT = 9999999

class TrafficLight:
    def __init__(self, tfID, algorithm='CDRL', yellow_duration=3, traci=None, cycle_control=5, config=None):
        self.id = tfID
        self.control_algorithm = algorithm
        self.yellow_duration = yellow_duration
        self.traci = traci
        self.cycle_control = cycle_control
        # traci.setOrder(2)

        self.lanes = self.traci.trafficlight.getControlledLanes(tfID)
        # self.lanes = []
        # Create controller based on the algorithm configed
        if self.control_algorithm == 'SOTL':
            self.controller = SOTL()
        elif self.control_algorithm == 'SimpleRL':
            self.controller = SimpleRL()
        elif self.control_algorithm == 'CDRL':
            self.controller = CDRL(config=config, tfID=tfID)
        elif self.control_algorithm == 'VFB':
            self.controller = VFB(config=config, tfID=tfID)
        elif self.control_algorithm == 'IntelliLight':
            self.controller = IntelliLight(config=config, tfID=tfID)
        else:
            print("Must implement method named %s" % algorithm)


        self.writer = tf.summary.create_file_writer('./logs/train/%s' % tfID)

        self.reset()

    def setLogic(self):
        """
            ### To do ###
            Set logic program for the traffic light
            Restart the logic at phase 0
        """
        self.traci.trafficlight.setPhase(self.id, 0)
        self.current_phase = 0
        self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)
    
    def reset(self):
        self.control_actions = []
        self.setLogic()
        self.last_action, self.last_state = None, None
        self.last_action_is_change = 0
        self.last_total_delay = 0
        self.last_list_veh = []
        self.action_type = self.controller.actionType()

    def getState(self):
        """
            return the current state of the intersection
        """
        all_logic_ = self.traci.trafficlight.getAllProgramLogics(self.id)[0]            
        current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
        
        lanes_unique_ = list(dict.fromkeys(self.lanes))
        vehs = []
        for lane in lanes_unique_:
            vehs.extend(self.traci.lane.getLastStepVehicleIDs(lane))

        return {'tfID': self.id, 'traci': self.traci, 'lanes': self.lanes, 'current_logic': current_logic, 
                'last_action_is_change': self.last_action_is_change, 'last_total_delay': self.last_total_delay, 'last_vehs': vehs}

    def update(self, is_train=False, pretrain=False):
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

        if len(self.control_actions) <= 0: 
            cur_state = self.getState()
            # if is training:
            #     check to explore
            if is_train:
                if pretrain or (random.uniform(0, 1) <= GlobalVariables.EXPLORE_PROBABILITY):
                    action = random.randint(0, 1)
                else:
                    action = self.controller.makeAction(cur_state)
                # log last_state, last_action, reward, cur_state
                if (self.last_state is not None) and (self.last_action is not None):
                    # compute reward
                    reward = self.controller.computeReward(cur_state)
                    self.controller.exp_memory.add([self.last_state, self.last_action, reward, self.controller.processState(cur_state)])
                    # plot reward
                    if not pretrain:
                        with self.writer.as_default():
                            tf.summary.scalar('reward', reward, step=GlobalVariables.step)

                self.last_state, self.last_action = self.controller.processState(cur_state), action
            else:
                action = self.controller.makeAction(cur_state)

            if self.action_type == ActionType.CHOICE_OF_PHASE:
                # handle action type of CHOICE_OF_PHASE:
                if cur_state['traci'].trafficlight.getPhase(cur_state['tfID']) != action:
                    to_change = 1
                else:
                    to_change = 0
            else:
                to_change = action

            # save measured performance of last state
            self.last_action_is_change = to_change

            # get total delay
            lanes = list(dict.fromkeys(self.lanes))
            vehs = []
            for lane in lanes:
                vehs.extend(cur_state['traci'].lane.getLastStepVehicleIDs(lane))
            total_delay = 0
            for veh in vehs:
                total_delay += 1 - cur_state['traci'].vehicle.getSpeed(veh) / cur_state['traci'].vehicle.getAllowedSpeed(veh)
            self.last_total_delay = total_delay
        
            if to_change == 1:
                current_phase_ = self.traci.trafficlight.getPhase(self.id)
                if current_phase_ < 3:
                    self.traci.trafficlight.setPhase(self.id, current_phase_ + 1)
                    self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)
                else:
                    print("******* error := change at yellow phase ??? ")
                    sys.exit()

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
                        sys.exit()

                    self.traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

        self.current_phase = self.traci.trafficlight.getPhase(self.id)

    def replay(self):
        self.controller.replay()