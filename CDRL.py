from RLAgent import RLAgent
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
np.set_printoptions(threshold=np.inf)

import math
import sumolib
import sys
from glo_vars import GloVars
from VFB import VFB
from controller import ActionType

class CDRL(RLAgent):
    def __init__(self, config, road_structure, number_of_phases):
        self.map_size, self.center_length_WE, self.center_length_NS = VFB.getMapSize(road_structure)
        RLAgent.__init__(self, config['cycle_control'], (self.map_size[0], self.map_size[1], 1), number_of_phases/2)

    @staticmethod
    def computeReward(state, historical_data):
        if historical_data == None:
            return 0

        reward = 0
        # penalty for change signal
        reward -= 0.1*historical_data['last_action_is_change']
        
        # get list vehicles
        vehs = state['vehicles']
        
        # penalty for teleports
        # num_veh_teleporting = 0
        # vehs_teleporting = traci.simulation.getStartingTeleportIDList()
        # for veh in vehs:
        #     if veh in vehs_teleporting:
        #         num_veh_teleporting += 1
        # reward -= 0.1*num_veh_teleporting
        
        # penalty for emergency stops
        # num_veh_emergency_stop = 0
        # vehs_emergency_stop = traci.simulation.getEmergencyStoppingVehiclesIDList()
        # for veh in vehs:
        #     if veh in vehs_emergency_stop:
        #         num_veh_emergency_stop += 1
        # reward -= 0.2*num_veh_emergency_stop

        # penalty for delay
        total_delay = 0
        for veh in vehs:
            total_delay += 1 - veh['speed'] / veh['max_speed']
        reward -= 0.3*total_delay

        # penalty for waiting time
        total_waiting_time = 0
        for veh in vehs:
            total_waiting_time += veh['waiting_time']
        reward -= 0.3*total_waiting_time

        return reward

    @staticmethod
    def buildModel(input_space, output_space):
        """
            return the model in keras
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(input_space)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_space, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def processState(self, state=None):
        """
            from general state returned from traffic light
            process to return position_map
        """
        position_map = VFB.buildMap(state, self.map_size, self.center_length_WE, self.center_length_NS)
        # add information on traffic light state

        if state['current_phase_index'] == 0:
            position_map[GloVars.ARRAY_LENGTH][GloVars.ARRAY_LENGTH], position_map[GloVars.ARRAY_LENGTH+self.center_length_NS][GloVars.ARRAY_LENGTH+self.center_length_WE] = 0.8, 0.8
            position_map[GloVars.ARRAY_LENGTH][GloVars.ARRAY_LENGTH + self.center_length_WE], position_map[GloVars.ARRAY_LENGTH+self.center_length_NS][GloVars.ARRAY_LENGTH] = 0.2, 0.2

        return np.reshape(position_map, (self.map_size[0], self.map_size[1], 1)) # reshape to (SPACE, 1)
    
    @staticmethod
    def logHistoricalData(state, action):
        historical_data = {}
        if action == ActionType.KEEP_PHASE:
            historical_data['last_action_is_change'] = 0
        else:
            historical_data['last_action_is_change'] = 1

        return historical_data