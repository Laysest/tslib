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
from VFB import VFB

class CAREL(RLAgent):
    def __init__(self, config, road_structure, number_of_phases):
        RLAgent.__init__(self, config['cycle_control'], int(number_of_phases/2), int(number_of_phases/2))

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
        max_q_length = CAREL.processState(state)
        r = np.sum(max_q_length) - np.sum(historical_data['max_q_length'])
        return r

    def processState(self, state):
        return CAREL.processState(state)

    @staticmethod
    def processState(state):
        from traffic_light import LightState

        phases = state['phase_description']
        vehs = state['vehicles']
        processed_state = []
        for idx, phase in enumerate(phases):
            if idx % 2 != 0:
                continue
            max_q_length = 0
            for link in phase:
                if link['light_state'] == LightState.Green:
                    q_length = 0
                    for veh in vehs:
                        if veh['lane'] == link['from'] and veh['speed'] < 5:
                            q_length += 1
                    if q_length > max_q_length:
                        q_length = max_q_length
            processed_state.append(max_q_length)
        
        return processed_state

    @staticmethod
    def logHistoricalData(state, action):
        return {'max_q_length': CAREL.processState(state)}