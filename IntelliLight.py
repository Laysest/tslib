from RLAgent import RLAgent
from controller import ActionType
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
np.set_printoptions(threshold=np.inf)

import math
import sumolib
import sys
import random
from glo_vars import GloVars
from VFB import VFB

# FEATURE_SPACE = (8*4)
PHASE_SPACE = (1)

# we only support 3-way and 4-way intersections
MAX_NUM_WAY = 4
# we assume that there are only 2 red/green phases, user can change this depend on their config
NUM_OF_RED_GREEN_PHASES = 2

class IntelliLight(RLAgent):
    def __init__(self, config=None, road_structure=None):    
        self.map_size, self.center_length_WE, self.center_length_NS = VFB.getMapSize(road_structure)
        self.incoming_lanes = [lane for k, road in road_structure.items() if 'in' in k for lane in road]
        self.outgoing_lanes = [lane for k, road in road_structure.items() if 'out' in k for lane in road]        
        RLAgent.__init__(self, config['cycle_control'])

    @staticmethod
    def computeReward(state, historical_data):
        if historical_data == None:
            return 0
        # penalty for queue lengths:
        road_structure = state['road_structure']
        vehs = state['vehicles']
        lanes = []
        for road in road_structure.keys():
            lanes.extend(road_structure[road])

        reward = 0
        # penalty for queuelength
        L = 0
        for lane in lanes:
            for veh in vehs:
                if veh['lane'] == lane['id'] and veh['speed'] < 5:
                    L += 1
        reward -= 0.25 * L

        # penalty for delay
        D = 0
        for lane in lanes:
            total_speed, count = 0, 0
            for veh in vehs:
                if veh['lane'] == lane['id']:
                    total_speed += veh['speed']
                    count += 1
            if count > 0:
                D += 1 - (total_speed/count)/lane['max_allowed_speed']
        reward -= 0.25*D

        # penalty for waiting time
        W = 0
        for lane in lanes:
            for veh in vehs:
                if veh['lane'] == lane['id']:
                    W += veh['waiting_time']
        W /= 60.0
        reward -= 0.25*W

        # penalty for change
        reward -= 5*historical_data['last_action_is_change']

        # reward for number vehicles passed
        N = 0
        vehs = state['vehicles']
        historical_vehs = state['vehicles']
        incoming_lanes_id = [lane['id'] for k, road in state['road_structure'].items() if 'in' in k for lane in road]

        for veh in historical_vehs:
            if veh['lane'] in incoming_lanes_id:
                tmp = next((item for item in vehs if item["id"] == veh["id"]), False)
                if (tmp == False) or (tmp['lane'] not in incoming_lanes_id):
                    N += 1

        # #reward for travel time of passed vehicles
        # total_travel_time = 0
        # for veh_id_ in vehs_id_passed:
        #     if (veh_id_ in GloVars.vehicles.keys()):
        #         veh_route_ = GloVars.vehicles[veh_id_].status['route']
        #         if len(veh_route_) >= 3:
        #             total_travel_time += veh_route_[-3]['last_time_step'] - veh_route_[-3]['first_time_step']
        # reward += total_travel_time/60

        return reward

    def buildModel(self):
        """
            return the model in keras
        """
        # model = Sequential()
        map_ = Input(shape=(self.map_size[0], self.map_size[1], 1))
        lane_features_ = Input(shape=4*len(self.incoming_lanes))
        # TODO -- FEATURE_SPACE depend on the intersection
        phase_ = Input(shape=PHASE_SPACE)
        
        conv1_ = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu")(map_)
        conv2_ = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation="relu")(conv1_)
        map_feature_ = Flatten()(conv2_)

        features_ = Concatenate()([lane_features_, map_feature_])
                  
        shared_hidden_1_ = Dense(64, activation='relu')(features_)

        separated_hidden_1_left_ = Dense(32, activation='relu')(shared_hidden_1_)
        output_left_ = Dense(GloVars.ACTION_SPACE, activation='linear')(separated_hidden_1_left_)

        separated_hidden_1_right_ = Dense(32, activation='relu')(shared_hidden_1_)
        output_right_ = Dense(GloVars.ACTION_SPACE, activation='linear')(separated_hidden_1_right_)

        out = tf.where(phase_ == 0, output_left_, output_right_)

        model = tf.keras.Model(inputs=[map_, lane_features_, phase_], outputs=out)
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def processState(self, state):
        """
            from general state returned from traffic light
            process to return position_map
        """
        map_ = VFB.buildMap(state, self.map_size, self.center_length_WE, self.center_length_NS)
        map_ = np.reshape(map_, (self.map_size[0], self.map_size[1], 1)) # reshape to (SPACE, 1)
        
        lane_features_ = self.getLaneFeatures(state)

        state_ = [np.array(map_), np.array(lane_features_), np.array([state['current_phase_index']])]
        return state_

    def getLaneFeatures(self, state):
        vehs = state['vehicles']
        # all_logic_ = traci.trafficlight.getAllProgramLogics(self.tf_id)[0]
        # current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
        # # lanes = list(dict.fromkeys(lanes_in_phases))
        lane_features_ = []

        # queue length
        for lane in self.incoming_lanes:
            queue_length = 0
            for veh in vehs:
                if veh['lane'] == lane['id'] and veh['speed'] < 5:
                    queue_length += 1
            lane_features_.append(queue_length)
        
        # waiting time
        for lane in self.incoming_lanes:
            waiting_time = 0
            for veh in vehs:
                if veh['lane'] == lane['id']:
                    waiting_time += veh['waiting_time']
            lane_features_.append(waiting_time)

        # phase vector
        for lane in self.incoming_lanes:
            for k, road in state['road_structure'].items():
                for sublane in road:
                    if sublane['id'] == lane['id']:
                        lane_features_.append(sublane['light_state'])
                        break

        # number of vehicles
        for lane in self.incoming_lanes:
            num_vehs = 0
            for veh in vehs:
                if veh['lane'] == lane['id']:
                    num_vehs += 1
            lane_features_.append(num_vehs)

        return lane_features_
    
    def actionType(self):
        return ActionType.CHANGING_KEEPING

    def replay(self):
        if self.exp_memory.len() < GloVars.SAMPLE_SIZE:
            return
        minibatch =  self.exp_memory.sample(GloVars.SAMPLE_SIZE)    
        batch_images = []
        batch_lane_features = []
        batch_phases = []
        batch_targets = []
        for state_, action_, reward_, next_state_ in minibatch:
            next_state_as_input_ = [np.array([next_state_[0]]), np.array([next_state_[1]]), next_state_[2]]
            qs = self.model.predict([next_state_as_input_])
            target = reward_ + GloVars.GAMMA*np.amax(qs[0])
            state_as_input_ = [np.array([state_[0]]), np.array([state_[1]]), state_[2]]
            target_f = self.model.predict(state_as_input_)
            target_f[0][action_] = target
            batch_images.append(state_[0])
            batch_lane_features.append(state_[1])
            batch_phases.append(state_[2][0])
            batch_targets.append(target_f[0])

        self.model.fit([np.array(batch_images), np.array(batch_lane_features), np.array(batch_phases)], np.array(batch_targets), 
                            epochs=GloVars.EPOCHS, batch_size=GloVars.BATCH_SIZE, shuffle=False, verbose=0, validation_split=0.3)

    def makeAction(self, state):
        state_ = self.processState(state)
        state_as_input_ = [np.array([state_[0]]), np.array([state_[1]]), state_[2]]
        out_ = self.model.predict(state_as_input_)[0]
        action = np.argmax(out_)

        if action == 1:
            return action, [{'type': ActionType.CHANGE_PHASE, 'length': self.cycle_control, 'executed': False}]
        return action, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]

    def randomAction(self, state):
        if random.randint(0, 1) == 0:
            return 0, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]
        return 1, [{'type': ActionType.CHANGE_PHASE, 'length': self.cycle_control, 'executed': False}]

    @staticmethod
    def logHistoricalData(state, action):
        historical_data = {}
        if action == ActionType.CHANGE_PHASE:
            historical_data['last_action_is_change'] = 1
        else:
            historical_data['last_action_is_change'] = 0
        historical_data['vehicles'] = state['vehicles']

        return historical_data