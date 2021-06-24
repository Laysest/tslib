import sys
import math
import sumolib
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from RLAgent import RLAgent
from glo_vars import GloVars
from controller import ActionType
from VFB import VFB

traci = GloVars.traci
# we only support 3-way and 4-way intersections
MAX_NUM_WAY = 4
# we assume that there are only 2 red/green phases, user can change this depend on their config
NUM_OF_RED_GREEN_PHASES = 2

ACTION_SPACE = GloVars.ACTION_SPACE * 2 + 1

NUM_TRAIN_STEP_TO_REPLACE = 2

class TLCC(RLAgent):
    def __init__(self, config=None, road_structure=None):
        self.map_size, self.center_length_WE, self.center_length_NS = VFB.getMapSize(road_structure)
        RLAgent.__init__(self, config['cycle_control'])
        self.q_net = self.model
        self.target_net = self.buildModel()
        self.phase_length = [self.cycle_control for _ in range(GloVars.ACTION_SPACE)]
        self.train_step = 0

    @staticmethod
    def computeReward(state, historical_data):
        return VFB.computeReward(state, historical_data)

    def buildModel(self):
        """
            return the model in keras
        """
        map_ = Input(shape=(self.map_size[0], self.map_size[1], 2))
        conv1_ = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu")(map_)
        conv2_ = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu")(conv1_)
        map_feature_ = Flatten()(conv2_)
        hidden_1_ = Dense(128, activation='relu')(map_feature_)
        hidden_2_v_ = Dense(64, activation='relu')(hidden_1_)
        hidden_2_a_ = Dense(64, activation='relu')(hidden_1_)
        v_ = Dense(1, activation='linear')(hidden_2_v_)
        a_ = Dense(ACTION_SPACE, activation='linear')(hidden_2_a_)
        Q_out_ = v_ + (a_ - tf.math.reduce_mean(a_, axis=1, keepdims=True))
        model = tf.keras.Model(inputs=map_, outputs=Q_out_)
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def processState(self, state=None):
        """
            from general state returned from traffic light
            process to return position_map
        """
        return np.reshape(TLCC.buildMap(state, self.map_size, self.center_length_WE, self.center_length_NS), (self.map_size[0], self.map_size[1], 2))

    @staticmethod
    def buildMap(state, map_size, center_length_WE, center_length_NS):
        """
            this function to return a 2D matrix indicating information on vehicles' positions
        """
        road_structure = state['road_structure']
        vehicles = state['vehicles']
        position_mapped = np.zeros(map_size)
        speed_mapped = np.zeros(map_size)

        # handle the North side
        if 'north_road_in' in road_structure:
            for i, lane in enumerate(road_structure['north_road_in']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=True)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j][GloVars.ARRAY_LENGTH + center_length_NS + i - len(road_structure['north_road_in'])] = arr_[j]
                    speed_mapped[j][GloVars.ARRAY_LENGTH + center_length_NS + i - len(road_structure['north_road_in'])] = spd_arr[j]
        if 'north_road_out' in road_structure:
            for i, lane in enumerate(road_structure['north_road_out']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=False)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j][GloVars.ARRAY_LENGTH + center_length_NS + len(road_structure['north_road_out']) - i - 1] = arr_[j]        
                    speed_mapped[j][GloVars.ARRAY_LENGTH + center_length_NS + len(road_structure['north_road_out']) - i - 1] = spd_arr[j]        

        # handle the East side
        if 'east_road_in' in road_structure:
            for i, lane in enumerate(road_structure['east_road_in']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=True)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_WE - len(road_structure['east_road_in']) + i][GloVars.ARRAY_LENGTH + center_length_NS + j + 1] = arr_[j]
                    speed_mapped[GloVars.ARRAY_LENGTH + center_length_WE - len(road_structure['east_road_in']) + i][GloVars.ARRAY_LENGTH + center_length_NS + j + 1] = spd_arr[j]
        if 'east_road_out' in road_structure:
            for i, lane in enumerate(road_structure['east_road_out']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=False)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_WE + len(road_structure['east_road_out']) - i - 1][GloVars.ARRAY_LENGTH + center_length_NS + j + 1] = arr_[j]
                    speed_mapped[GloVars.ARRAY_LENGTH + center_length_WE + len(road_structure['east_road_out']) - i - 1][GloVars.ARRAY_LENGTH + center_length_NS + j + 1] = spd_arr[j]

        # handle the South side
        if 'south_road_in' in road_structure:
            for i, lane in enumerate(road_structure['south_road_in']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=True)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j + GloVars.ARRAY_LENGTH + center_length_WE + 1][GloVars.ARRAY_LENGTH + center_length_NS + len(road_structure['south_road_in']) - i - 1] = arr_[j]
                    speed_mapped[j + GloVars.ARRAY_LENGTH + center_length_WE + 1][GloVars.ARRAY_LENGTH + center_length_NS + len(road_structure['south_road_in']) - i - 1] = spd_arr[j]

        if 'south_road_out' in road_structure:
            for i, lane in enumerate(road_structure['south_road_out']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=False)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j + GloVars.ARRAY_LENGTH + center_length_WE + 1][GloVars.ARRAY_LENGTH + center_length_NS - len(road_structure['south_road_out']) + i] = arr_[j]
                    speed_mapped[j + GloVars.ARRAY_LENGTH + center_length_WE + 1][GloVars.ARRAY_LENGTH + center_length_NS - len(road_structure['south_road_out']) + i] = spd_arr[j]

        # handle the West side
        if 'west_road_in' in road_structure:
            for i, lane in enumerate(road_structure['west_road_in']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=True)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_WE + len(road_structure['west_road_in']) - i - 1][j] = arr_[j]
                    speed_mapped[GloVars.ARRAY_LENGTH + center_length_WE + len(road_structure['west_road_in']) - i - 1][j] = spd_arr[j]

        if 'west_road_out' in road_structure:
            for i, lane in enumerate(road_structure['west_road_out']):
                arr_, spd_arr = TLCC.buildArray(lane=lane, vehicles=vehicles, incoming=False)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_WE - len(road_structure['west_road_out']) + i][j] = arr_[j]
                    speed_mapped[GloVars.ARRAY_LENGTH + center_length_WE - len(road_structure['west_road_out']) + i][j] = spd_arr[j]

        return [position_mapped, speed_mapped]
    
    @staticmethod
    def buildArray(lane=None, vehicles=None, incoming=None):
        pos_arr = np.zeros(GloVars.ARRAY_LENGTH)
        spd_arr = np.zeros(GloVars.ARRAY_LENGTH)
        # lane = 'CtoW_0', 'EtoC_0' It is inverted for this case
        vehs = [veh for veh in vehicles if veh['lane'] == lane['id']]
        for veh in vehs:
            veh_distance = veh['distance_from_lane_start']
            if incoming:
                veh_distance -= lane['length'] - GloVars.LENGTH_CELL*GloVars.ARRAY_LENGTH
            if veh_distance < 0:
                continue
            index = math.floor(veh_distance/5)

            if index >= GloVars.ARRAY_LENGTH:
                continue

            for i in range(math.ceil(veh['length']/5)):
                if 0 <= index - i < GloVars.ARRAY_LENGTH:
                    pos_arr[index - i] = 1
                    spd_arr[index - i] = veh['speed']
        return pos_arr, spd_arr

    def replay(self):
        if self.exp_memory.len() < GloVars.SAMPLE_SIZE:
            return
        mini_batch = self.exp_memory.sample(GloVars.SAMPLE_SIZE)
        state_, action_, reward_, next_state_ = zip(*mini_batch)
        target = self.q_net.predict(np.array(state_))
        next_state_val = self.target_net.predict(np.array(next_state_))
        best_action_index = np.argmax(self.q_net.predict(np.array(next_state_)), axis=1)
        batch_index = np.arange(GloVars.BATCH_SIZE, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, action_] = reward_ + GloVars.GAMMA * next_state_val[batch_index, best_action_index]        
        self.q_net.fit(np.array(state_), np.array(q_target), epochs=GloVars.EPOCHS, batch_size=GloVars.BATCH_SIZE, shuffle=False, verbose=0, validation_split=0.3)
        self.train_step += 1
        if self.train_step > 0 and self.train_step % 2 == 0:
            self.updateTargetNet()

    def updateTargetNet(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def randomAction(self, state):
        action = random.randint(0, ACTION_SPACE - 1)
        if action == 0:
            self.phase_length[0] += self.cycle_control
        elif action == 1:
            self.phase_length[1] += self.cycle_control
        elif action == 2:
            self.phase_length[1] -= self.cycle_control
        elif action == 3:
            self.phase_length[0] -= self.cycle_control
        self.limitPhaseLength()

        return action, [{'type': ActionType.CHANGE_PHASE, 'length': self.phase_length[0], 'executed': False},
                    {'type': ActionType.CHANGE_PHASE, 'length': self.phase_length[1], 'executed': False}]

    def makeAction(self, state):
        state_ = self.processState(state)
        out_ = self.model.predict(np.array([state_]))[0]
        action = np.argmax(out_)

        if action == 0:
            self.phase_length[0] += self.cycle_control
        elif action == 1:
            self.phase_length[1] += self.cycle_control
        elif action == 2:
            self.phase_length[1] -= self.cycle_control
        elif action == 3:
            self.phase_length[0] -= self.cycle_control
        self.limitPhaseLength()

        return action, [{'type': ActionType.CHANGE_PHASE, 'length': self.phase_length[0], 'executed': False},
                    {'type': ActionType.CHANGE_PHASE, 'length': self.phase_length[1], 'executed': False}]

    def limitPhaseLength(self):
        # Limit phase_length
        if self.phase_length[0] > 60:
            self.phase_length[0] = 60
        elif self.phase_length[0] < 0:
            self.phase_length[0] = 0
        if self.phase_length[1] > 60:
            self.phase_length[1] = 60
        elif self.phase_length[0] < 0:
            self.phase_length[0] = 0

    @staticmethod
    def logHistoricalData(state, action):
        historical_data = {}
        total_delay = 0
        for veh in state['vehicles']:
            total_delay += 1 - veh['speed'] / veh['max_speed']
        historical_data['last_total_delay'] = total_delay
        return historical_data