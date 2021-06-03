"""
    This file declare TrafficLight class
"""
import random
import sys
import os
import tensorflow as tf
from SOTL import SOTL
from SimpleRL import SimpleRL
from CDRL import CDRL
from VFB import VFB
from IntelliLight import IntelliLight
from TLCC import TLCC
from FixedTime import FixedTime
from MaxPressure import MaxPressure
from glo_vars import GloVars
from controller import ActionType, Controller


MAX_INT = 9999999
traci = GloVars.traci

# TODO: change it to more general
MAX_NUM_PHASE = 4

class TrafficLight:
    # pylint: disable=line-too-long invalid-name too-many-instance-attributes
    '''
        TrafficLight for each intersection having traffic signal control
    '''
    def __init__(self, config=None):
        self.id = config['node_id']
        self.control_algorithm = config['method']
        self.yellow_duration = config['yellow_duration']
        self.cycle_control = config['cycle_control']
        self.folder = config['folder']
        self.lanes = traci.trafficlight.getControlledLanes(self.id)
        # self.lanes = []
        # Create controller based on the algorithm configed
        if self.control_algorithm == 'SOTL':
            self.controller = SOTL(config=config, tf_id=self.id)
        elif self.control_algorithm == 'CDRL':
            self.controller = CDRL(config=config, tf_id=self.id)
        elif self.control_algorithm == 'VFB':
            self.controller = VFB(config=config, tf_id=self.id)
        elif self.control_algorithm == 'IntelliLight':
            self.controller = IntelliLight(config=config, tf_id=self.id)
        elif self.control_algorithm == 'TLCC':
            self.controller = TLCC(config=config, tf_id=self.id)
        elif self.control_algorithm == 'FixedTime':
            self.controller = FixedTime(config=config, tf_id=self.id)
        elif self.control_algorithm == 'MaxPressure':
            self.controller = MaxPressure(config=config, tf_id=self.id)
        else:
            print("Must implement method named %s" % self.control_algorithm)

        self.writer = tf.summary.create_file_writer('./tensorboard/atlanta/%s-%s' % (self.id, self.control_algorithm))
        self.current_phase = 0
        self.last_action, self.last_processed_state, self.last_state = None, None, None
        self.last_action_is_change = 0
        self.last_total_delay = 0
        self.lanes = traci.trafficlight.getControlledLanes(self.id)
        self.lanes_unique = list(dict.fromkeys(self.lanes))
        self.reset()

         # this for  computing reward
        self.historical_data = None
        self.log = {
            'lanes': self.lanes_unique
        }

    def log_step(self):
        now = traci.simulation.getTime()
        self.log[now] = {
            'CO2_emission': [traci.lane.getCO2Emission(lane) for lane in self.lanes_unique],
            'CO_emission': [traci.lane.getCOEmission(lane) for lane in self.lanes_unique],
            'fuel_consumption': [traci.lane.getFuelConsumption(lane) for lane in self.lanes_unique],
            'num_halting_vehs': [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes_unique],
            'speed': [traci.lane.getLastStepMeanSpeed(lane) for lane in self.lanes_unique],
            'occupancy': [traci.lane.getLastStepOccupancy(lane) for lane in self.lanes_unique],
            'num_vehs': [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes_unique],
            'waiting_time': [traci.lane.getWaitingTime(lane) for lane in self.lanes_unique],
            'queue_length': self.getQueueLength()
        }
    
    def getQueueLength(self):
        queue_length = [];
        for lane in self.lanes_unique:
            q = 0
            vehs = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehs:
                if traci.vehicle.getSpeed(veh) < 5:
                    q += traci.vehicle.getLength(veh)
            queue_length.append(q)
        return queue_length
        
    def setLogic(self):
        """
            ### To do ###
            Set logic program for the traffic light
            Restart the logic at phase 0
        """
        traci.trafficlight.setPhase(self.id, 0)
        self.current_phase = 0
        traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

    def reset(self):
        if self.control_algorithm != 'FixedTime':
            self.setLogic()
        self.control_actions = []
        self.last_action, self.last_processed_state, self.last_state = None, None, None
        self.last_action_is_change = 0
        self.last_total_delay = 0
        self.last_list_veh = []
        self.historical_data = {
            'CDRL': {
                'last_action_is_change': 0
            },
            'VFB': {
                'last_total_delay': 0
            },
            'IntelliLight':{
                'last_action_is_change': 0,
                'last_vehs_id': []
            }
        }
        self.log = {
            'lanes': self.lanes_unique
        }

    def getState(self):
        """
            return the current state of the intersection
        """
        all_logic_ = traci.trafficlight.getAllProgramLogics(self.id)[0]
        current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
        current_phase_index = traci.trafficlight.getPhase(self.id)
        lanes_unique_ = list(dict.fromkeys(self.lanes))
        vehs_id = []
        for lane in lanes_unique_:
            vehs_id.extend(traci.lane.getLastStepVehicleIDs(lane))

        return {'tf_id': self.id, 'lanes': self.lanes, 'current_logic': current_logic, 'current_phase_index': current_phase_index,
                'last_action_is_change': self.last_action_is_change, 'last_total_delay': self.last_total_delay, 'vehs_id': vehs_id}

    def processControlStack(self, control_stack):
        if len(control_stack) <= 0:
            return
        for action in control_stack:
            if action['type'] == ActionType.CHANGE_PHASE:
                if action['length'] > 0:
                    self.control_actions.extend([{'type': ActionType.YELLOW_PHASE, 'length': self.yellow_duration, 'executed': False},
                                                {'type': ActionType.CHANGE_PHASE, 'length': action['length'], 'executed': False}])
                else:
                    # TODO if action['length] == 0:
                    self.control_actions.extend([{'type': ActionType.YELLOW_PHASE, 'length':self.yellow_duration, 'executed': False}, # change to yellow phase
                                                {'type': ActionType.CHANGE_PHASE, 'length':  0, 'executed': False}, # change to next phase
                                                {'type': ActionType.CHANGE_PHASE, 'length': 0, 'executed': False},  # change to yellow phase
                                                {'type': ActionType.CHANGE_PHASE, 'length': 0, 'executed': False}]) # change to next of next phase

                    # self.control_actions.extend([{'type': ActionType.YELLOW_PHASE, 'length': self.yellow_duration, 'executed': False},
                    #                             {'type': ActionType.CHANGE_PHASE, 'length': action['length'], 'executed': False}])
            elif action['type'] == ActionType.KEEP_PHASE:
                self.control_actions.extend([{'type': ActionType.KEEP_PHASE, 'length': action['length'], 'executed': False}])
            else:
                print("error in processControlStack")
                sys.exit()

    def changeToNextPhase(self):
        """
            Call this function to change the traffic light to the next phase
        """
        current_phase_ = traci.trafficlight.getPhase(self.id)
        if current_phase_ >= MAX_NUM_PHASE - 1:
            traci.trafficlight.setPhase(self.id, 0)
        else:
            traci.trafficlight.setPhase(self.id, current_phase_ + 1)
        traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

    def doAction(self):
        """
            This function is to execute the control_actions
            if not executed => executed
            -- length
            if length = 0 => pop
        """
        while len(self.control_actions) > 0:
            if self.control_actions[0]['executed'] is False:
                self.control_actions[0]['executed'] = True
                if self.control_actions[0]['type'] == ActionType.CHANGE_PHASE or self.control_actions[0]['type'] == ActionType.YELLOW_PHASE:
                    self.changeToNextPhase()
            if self.control_actions[0]['length'] > 1:
                self.control_actions[0]['length'] -= 1
                break
            elif self.control_actions[0]['length'] == 1:
                self.control_actions[0]['length'] -= 1
                self.control_actions.pop(0)
                break
            elif self.control_actions[0]['length'] == 0:
                self.control_actions.pop(0)
                continue
            else:
                print("error in process control stack")
                sys.exit()              
        # if len(self.control_actions) > 0:
        #     if self.control_actions[0]['executed'] is False:
        #         self.control_actions[0]['executed'] = True
        #         if self.control_actions[0]['type'] == ActionType.CHANGE_PHASE or self.control_actions[0]['type'] == ActionType.YELLOW_PHASE:
        #             self.changeToNextPhase()
        #     self.control_actions[0]['length'] -= 1
        #     if self.control_actions[0]['length'] <= 0:
        #         self.control_actions.pop(0)

    def logHistoricalData(self, last_action):
        if last_action == ActionType.CHANGE_PHASE:
            self.historical_data['CDRL']['last_action_is_change'] = 1
            self.historical_data['IntelliLight']['last_action_is_change'] = 1
        else:
            self.historical_data['CDRL']['last_action_is_change'] = 0
            self.historical_data['IntelliLight']['last_action_is_change'] = 0

        vehs = []
        for lane in self.lanes_unique:
            vehs.extend(traci.lane.getLastStepVehicleIDs(lane))
        self.historical_data['IntelliLight']['last_vehs_id'] = vehs

        # total delay
        total_delay = 0
        for veh in vehs:
            total_delay += 1 - traci.vehicle.getSpeed(veh) / traci.vehicle.getAllowedSpeed(veh)
        self.historical_data['VFB']['last_total_delay'] = total_delay

    def saveModel(self, ep=0):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.controller.model.save("%s/%s-%d.h5" % (self.folder, self.id, ep))
    
    def loadModel(self):
        try:
            #TODO: take care with 49
            self.controller.model.load_weights("%s/%s-%d.h5" % (self.folder, self.id, 49))
        except:
            print("No model to load %s/%s-%d.h5" % (self.folder, self.id, 49))

    def update(self, is_train=False, pretrain=False):
        """
            Call this function each time step
        """
        # TODO - new update function to suit for a stack of actions and more phases of a cycle
        # ....
        # every changing action has a yellow phase following
        # control_actions = [{'type': '1', 'length': 3, executed: True/False}, {'type': '1', 'length': 5, executed: True/False}]
        #                    => change -> keep this phase (yellow phase) for 3 seconds        => change -> keep this phase (green/red) for 5 seconds

        #   if len = 0 => no action in queue:
        #       get action by state
        #       process action & add into the control_actions
        #   if len >= 0 => have action in queue:
        #       if not yet execute => execute by +1 phase index
        #       -1 time length
        #           if time length == 0:
        #               that means finish the cycle => delete in queue

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
        control_stack = []
        if len(self.control_actions) <= 0:
            cur_state = self.getState()
            if is_train:
                if pretrain or (random.uniform(0, 1) <= GloVars.EXPLORE_PROBABILITY):
                    action, control_stack = self.controller.randomAction(cur_state)
                else:
                    action, control_stack = self.controller.makeAction(cur_state)

                # log last_state, last_action, reward, cur_state
                if (self.last_processed_state is not None) and (self.last_action is not None):
                    # compute reward
                    reward = self.controller.computeReward(cur_state, self.historical_data)
                    self.logHistoricalData(control_stack[0]['type'])
                    self.controller.exp_memory.add([self.last_processed_state, self.last_action, reward, self.controller.processState(cur_state)])
                    # plot reward
                    if not pretrain:
                        with self.writer.as_default():
                            tf.summary.scalar('reward', reward, step=GloVars.step)

                self.last_processed_state, self.last_action = self.controller.processState(cur_state), action
            else:
                action, control_stack = self.controller.makeAction(cur_state)

        self.processControlStack(control_stack)
        self.doAction()

            # TODO - log last state for computing reward
            # self.last_action_is_change =         ??

        # if len(self.control_actions) <= 0:
        #     cur_state = self.getState()
        #     # if is training:
        #     #     check to explore
        #     if is_train:
        #         if pretrain or (random.uniform(0, 1) <= GloVars.EXPLORE_PROBABILITY):
        #             action, stack_controls = self.controller.randomAction(cur_state)
        #         else:
        #             action, stack_controls = self.controller.makeAction(cur_state)
        #         # log last_state, last_action, reward, cur_state
        #         if (self.last_processed_state is not None) and (self.last_action is not None) and (self.last_state is not None):
        #             # compute reward
        #             reward = self.controller.computeReward(cur_state, self.last_state)
        #             self.controller.exp_memory.add([self.last_processed_state, self.last_action, reward, self.controller.processState(cur_state)])
        #             # plot reward
        #             if not pretrain:
        #                 with self.writer.as_default():
        #                     tf.summary.scalar('reward', reward, step=GloVars.step)

        #         self.last_state = cur_state
        #         self.last_processed_state, self.last_action = self.controller.processState(cur_state), action
        #     else:
        #         action, stack_controls = self.controller.makeAction(cur_state)

        #     if self.action_type == ActionType.CHOICE_OF_PHASE:
        #         # handle action type of CHOICE_OF_PHASE:
        #         if traci.trafficlight.getPhase(cur_state['tf_id']) != action:
        #             to_change = 1
        #         else:
        #             to_change = 0
        #     else:
        #         to_change = action

        #     # save measured performance of last state
        #     self.last_action_is_change = to_change

        #     # get total delay -----------------------------------------------------------------------
        #     lanes = list(dict.fromkeys(self.lanes))
        #     vehs = []
        #     for lane in lanes:
        #         vehs.extend(traci.lane.getLastStepVehicleIDs(lane))
        #     total_delay = 0
        #     for veh in vehs:
        #         total_delay += 1 - traci.vehicle.getSpeed(veh) / traci.vehicle.getAllowedSpeed(veh)
        #     self.last_total_delay = total_delay
        #     # ---------------------------------------------------------------------------------------

        #     if to_change == 1:
        #         current_phase_ = traci.trafficlight.getPhase(self.id)
        #         if current_phase_ < 3:
        #             traci.trafficlight.setPhase(self.id, current_phase_ + 1)
        #             self.control_actionstraci.trafficlight.setPhaseDuration(self.id, MAX_INT)
        #         else:
        #             print("******* error := change at yellow phase ??? ")
        #             sys.exit()

        #         self.control_actions.extend([{'type': 'yellow_phase', 'length': self.yellow_duration},
        #                                    {'type': 'red_green_phase', 'length': self.cycle_control}])
        #     else:
        #         self.control_actions.append({'type': 'red_green_phase', 'length': self.cycle_control})

        # if len(self.control_actions) > 0:
        #     self.control_actions[0]['length'] -= 1
        #     if self.control_actions[0]['length'] <= 0:
        #         self.control_actions.pop(0)
        #         if len(self.control_actions) > 0:
        #             current_phase_ = traci.trafficlight.getPhase(self.id)
        #             if current_phase_ == 1:
        #                 traci.trafficlight.setPhase(self.id, current_phase_ + 1)
        #             elif current_phase_ == 3:
        #                 traci.trafficlight.setPhase(self.id, 0)
        #             else:
        #                 print("******* error in control: current_phase: %d, control_actions: %s" % (current_phase_, self.control_actions))
        #                 sys.exit()
        #             traci.trafficlight.setPhaseDuration(self.id, MAX_INT)

        # self.current_phase = traci.trafficlight.getPhase(self.id)

    def replay(self):
        self.controller.replay()
