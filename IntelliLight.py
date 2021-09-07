from RLAgent import RLAgent
from controller import ActionType
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
import sumolib
import sys
import random
from glo_vars import GloVars
from VFB import VFB

class IntelliLight(RLAgent):
    def __init__(self, config, road_structure, number_of_phases):
        self.map_size, self.center_length_WE, self.center_length_NS = VFB.getMapSize(road_structure)
        self.incoming_lanes = [lane for k, road in road_structure.items() if 'in' in k for lane in road]
        self.outgoing_lanes = [lane for k, road in road_structure.items() if 'out' in k for lane in road]

        input_space = ((self.map_size[0], self.map_size[1], 1), len(self.incoming_lanes)*4)
        RLAgent.__init__(self, config['cycle_control'], input_space, int(number_of_phases/2))
        
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
    
    @staticmethod
    def buildModel(input_space, output_space):
        """
            return the model in keras
        """
        # model = Sequential()
        map_size = input_space[0]
        feature_length = input_space[1]

        map_ = Input(shape=map_size)
        lane_features_ = Input(shape=feature_length)
        phase_ = Input(shape=(1))
        
        conv1_ = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu")(map_)
        conv2_ = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation="relu")(conv1_)
        map_feature_ = Flatten()(conv2_)

        features_ = Concatenate()([lane_features_, map_feature_])
                  
        shared_hidden_ = Dense(64, activation='relu')(features_)

        output = []
        for i in range(output_space):
            separated_hidden_ = Dense(32, activation='relu')(shared_hidden_)
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
        from traffic_light import LightState
        vehs = state['vehicles']
        # all_logic_ = traci.trafficlight.getAllProgramLogics(self.tfl_id)[0]
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
        light_state = {}
        phase_detail = state['phase_description'][state['current_phase_index']]

        for item in phase_detail:
            if item['from'] not in light_state:
                light_state[item['from']] = 0
            if item['light_state'] == LightState.Green:
                light_state[item['from']] += 1

        for lane in self.incoming_lanes:
            if lane['id'] in light_state.keys():
                if light_state[lane['id']] > 0:
                    lane_features_.append(LightState.Green)
                else:
                    lane_features_.append(LightState.Red)                    
            else:
                lane_features_.append(LightState.Red)

        # number of vehicles
        for lane in self.incoming_lanes:
            num_vehs = 0
            for veh in vehs:
                if veh['lane'] == lane['id']:
                    num_vehs += 1
            lane_features_.append(num_vehs)

        return lane_features_
    
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
            return action, [{'type': ActionType.CHANGE_TO_NEXT_PHASE, 'length': self.cycle_control, 'executed': False}]
        return action, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]

    def randomAction(self, state):
        if random.randint(0, 1) == 0:
            return 0, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]
        return 1, [{'type': ActionType.CHANGE_TO_NEXT_PHASE, 'length': self.cycle_control, 'executed': False}]

    @staticmethod
    def logHistoricalData(state, action):
        historical_data = {}
        if action == ActionType.KEEP_PHASE:
            historical_data['last_action_is_change'] = 0
        else:
            historical_data['last_action_is_change'] = 1
        historical_data['vehicles'] = state['vehicles']

        return historical_data