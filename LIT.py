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

class LIT(RLAgent):
    def __init__(self, config, road_structure, number_of_phases):
        self.incoming_lanes = [lane for k, road in road_structure.items() if 'in' in k for lane in road]
        RLAgent.__init__(self, config['cycle_control'], len(self.incoming_lanes), 2)

    @staticmethod
    def buildModel(input_space, output_space):
        """
            return the model in keras
        """
        # model = Sequential()
        feature_length = input_space

        lane_features_ = Input(shape=feature_length)
        phase_ = Input(shape=(1))
                      
        shared_hidden_ = Dense(16, activation='relu')(lane_features_)

        output = []
        for i in range(output_space):
            separated_hidden_ = Dense(16, activation='relu')(shared_hidden_)
            output_ = Dense(output_space, activation='linear')(separated_hidden_)
            output.append(output_)

        if output_space == 2:
            out = tf.where(phase_ == 0, output[0], output[1])
        elif output_space == 3:
            out = tf.where(phase_ == 0, output[0], tf.where(phase_ == 1, output[1], output[2]))
        elif output_space == 4:
            out = tf.where(phase_ == 0, output[0], tf.where(phase_ == 1, output[1], tf.where(phase_ == 2, output[2], output[3])))
        elif output_space == 5:
            out = tf.where(phase_ == 0, output[0], tf.where(phase_ == 1, output[1], tf.where(phase_ == 2, output[2], tf.where(phase_ == 3, output[3], output[4]))))
        elif output_space == 6:
            out = tf.where(phase_ == 0, output[0], tf.where(phase_ == 1, output[1], tf.where(phase_ == 2, output[2], tf.where(phase_ == 3, output[3], tf.where(phase_ == 4, output[4], output[5])))))
        else:
            print("<<< IntelliLight support only maximum 6 phases >>>")
            sys.exit(0)

        model = tf.keras.Model(inputs=[lane_features_, phase_], outputs=out)
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    @staticmethod
    def computeReward(state, historical_data):
        L = 0
        incoming_lanes = [lane for k, road in state['road_structure'].items() if 'in' in k for lane in road]
        for lane in incoming_lanes:
            for veh in state['vehicles']:
                if veh['lane'] == lane['id'] and veh['speed'] < 5:
                    L -= 1
        return L

    def processState(self, state):
        arr = [0] * len(self.incoming_lanes)
        for idx, lane in enumerate(self.incoming_lanes):
            for veh in state['vehicles']:
                if veh['lane'] == lane['id']:
                    arr[idx] += 1
        
        return [np.array(arr), np.array([state['current_phase_index']])]


    def makeAction(self, state):
        state_ = self.processState(state)
        state_as_input_ = [np.array([state_[0]]), np.array([state_[1]])]
        out_ = self.model.predict(state_as_input_)[0]
        action = np.argmax(out_)

        if action == 1:
            return action, [{'type': ActionType.CHANGE_TO_NEXT_PHASE, 'length': self.cycle_control, 'executed': False}]
        return action, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]

    def randomAction(self, state):
        if random.randint(0, 1) == 0:
            return 0, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]
        return 1, [{'type': ActionType.CHANGE_TO_NEXT_PHASE, 'length': self.cycle_control, 'executed': False}]

    def replay(self):
        if self.exp_memory.len() < GloVars.SAMPLE_SIZE:
            return
        minibatch =  self.exp_memory.sample(GloVars.SAMPLE_SIZE)    
        batch_lane_features = []
        batch_phases = []
        batch_targets = []
        for state_, action_, reward_, next_state_ in minibatch:
            next_state_as_input_ = [np.array([next_state_[0]]), np.array([next_state_[1]])]
            qs = self.model.predict([next_state_as_input_])
            target = reward_ + GloVars.GAMMA*np.amax(qs[0])
            state_as_input_ = [np.array([state_[0]]), np.array([state_[1]])]
            target_f = self.model.predict(state_as_input_)
            target_f[0][action_] = target
            batch_lane_features.append(state_[1])
            batch_phases.append(state_[2][0])
            batch_targets.append(target_f[0])

        self.model.fit([np.array(batch_lane_features), np.array(batch_phases)], np.array(batch_targets), 
                            epochs=GloVars.EPOCHS, batch_size=GloVars.BATCH_SIZE, shuffle=False, verbose=0, validation_split=0.3)

    @staticmethod
    def logHistoricalData(state, action):
        return {}