from RLAgent import RLAgent
from controller import ActionType
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
import sumolib
import sys
import random
from glo_vars import GloVars

class LITP(RLAgent):
    def __init__(self, config, road_structure, number_of_phases):
        self.incoming_lanes = [lane for k, road in road_structure.items() if 'in' in k for lane in road]
        RLAgent.__init__(self, config['cycle_control'], len(self.incoming_lanes), int(number_of_phases/2))

    @staticmethod
    def buildModel(input_space, output_space):
        """
            return the model in keras
        """
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=input_space))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(output_space, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    @staticmethod
    def computeReward(state, historical_data):
        if historical_data == None:
            return 0
        incoming_lanes = [lane for k, road in state['road_structure'].items() if 'in' in k for lane in road]
        q_length_arr = LITP.getQLength(state['vehicles'], incoming_lanes)
        r = np.sum(q_length_arr)
        return r * -1

    def processState(self, state):
        return LITP.getQLength(state['vehicles'], self.incoming_lanes)

    @staticmethod
    def getQLength(vehs, incoming_lanes):
        q_length_arr = []
        for lane in incoming_lanes:
            q_length = 0
            for veh in vehs:
                if veh['lane'] == lane['id'] and veh['speed'] < 5:
                    q_length += 1
            q_length_arr.append(q_length)
        
        return q_length_arr


    @staticmethod
    def logHistoricalData(state, action):
        return {}