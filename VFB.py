import sys
import math
import sumolib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from RLAgent import RLAgent
from glo_vars import GloVars

np.set_printoptions(threshold=np.inf)
traci = GloVars.traci
# we only support 3-way and 4-way intersections
MAX_NUM_WAY = 4

class VFB(RLAgent):
    def __init__(self, config=None, tf_id=None, road_structure=None):
        self.map_size, self.center_length_WE, self.center_length_NS = self.getMapSize(road_structure)
        RLAgent.__init__(self, config['cycle_control'])
        self.config = config
        self.tf_id = tf_id

    def getMapSize(self, road_structure):
        center_length_NS = 1
        if 'east_road_in' in road_structure:
            center_length_NS = len(road_structure['east_road_in'])
        if ('west_road_in' in road_structure) and (center_length_NS < len(road_structure['west_road_in'])):
            center_length_NS = len(road_structure['west_road_in'])

        center_length_WE = 1
        if 'north_road_in' in road_structure:
            center_length_WE = len(road_structure['north_road_in'])
        if ('south_road_in' in road_structure) and (center_length_WE < len(road_structure['south_road_in'])):
            center_length_WE = len(road_structure['south_road_in'])

        map_size = (2*(GloVars.ARRAY_LENGTH + center_length_WE), 2*(GloVars.ARRAY_LENGTH + center_length_NS))
        return map_size, center_length_WE, center_length_NS

    def computeReward(self, state, historical_data):
        reward = 0

        # get list vehicles
        vehs = []
        for lane in self.lanes_unique:
            vehs.extend(traci.lane.getLastStepVehicleIDs(lane))
        
        # total delay
        total_delay = 0
        for veh in vehs:
            total_delay += 1 - traci.vehicle.getSpeed(veh) / traci.vehicle.getAllowedSpeed(veh)
        
        reward = historical_data['VFB']['last_total_delay'] - total_delay

        return reward

    def buildModel(self):
        """
            return the model in keras
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.map_size[0], self.map_size[1], 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(GloVars.ACTION_SPACE, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def processState(self, state=None):
        """
            from general state returned from traffic light
            process to return position_map
        """
        map_ = self.buildMap(state)
        return np.reshape(map_, (self.map_size[0], self.map_size[1], 1)) # reshape to (SPACE, 1)

    def buildMap(self, state):
        """
            this function to return a 2D matrix indicating information on vehicles' positions
        """
        road_structure = state['road_structure']
        vehicles = state['vehicles']
        position_mapped = np.zeros(self.map_size)
        center_length_WE, center_length_NS = self.center_length_WE, self.center_length_NS

        # handle the North side
        if 'north_road_in' in road_structure:
            for i, lane in enumerate(road_structure['north_road_in']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=True)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j][GloVars.ARRAY_LENGTH + center_length_WE + i - len(road_structure['north_road_in'])] = arr_[j]
        if 'north_road_out' in road_structure:
            for i, lane in enumerate(road_structure['north_road_out']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=False)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j][GloVars.ARRAY_LENGTH + center_length_WE + len(road_structure['north_road_out']) - i - 1] = arr_[j]        

        # handle the East side
        if 'east_road_in' in road_structure:
            for i, lane in enumerate(road_structure['east_road_in']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=True)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_NS - len(road_structure['east_road_in']) + i][GloVars.ARRAY_LENGTH + center_length_WE + j + 1] = arr_[j]
        if 'east_road_out' in road_structure:
            for i, lane in enumerate(road_structure['east_road_out']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=False)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_NS + len(road_structure['east_road_out']) - i - 1][GloVars.ARRAY_LENGTH + center_length_WE + j + 1] = arr_[j]

        # handle the South side
        if 'south_road_in' in road_structure:
            for i, lane in enumerate(road_structure['south_road_in']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=True)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j + GloVars.ARRAY_LENGTH + center_length_NS + 1][GloVars.ARRAY_LENGTH + center_length_WE + len(road_structure['south_road_in']) - i - 1] = arr_[j]
        if 'south_road_out' in road_structure:
            for i, lane in enumerate(road_structure['south_road_out']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=False)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j + GloVars.ARRAY_LENGTH + center_length_NS + 1][GloVars.ARRAY_LENGTH + center_length_WE - len(road_structure['south_road_out']) + i] = arr_[j]

        # handle the West side
        if 'west_road_in' in road_structure:
            for i, lane in enumerate(road_structure['west_road_in']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=True)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_NS + len(road_structure['west_road_in']) - i - 1][j] = arr_[j]
        if 'west_road_out' in road_structure:
            for i, lane in enumerate(road_structure['west_road_out']):
                arr_ = self.buildArray(lane=lane, vehicles=vehicles, incoming=False)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + center_length_NS - len(road_structure['west_road_out']) + i][j] = arr_[j]

        print(position_mapped)
        return position_mapped
    
    def buildArray(self, lane=None, vehicles=None, incoming=None):
        arr = np.zeros(GloVars.ARRAY_LENGTH)
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
                    arr[index - i] = 1

        return arr

    def logHistoricalData(self, state, action):
        historical_data = {}
        total_delay = 0
        for veh in state['vehicles']:
            total_delay += 1 - veh['speed'] / veh['max_speed']
        historical_data['VFB']['last_total_delay'] = total_delay
        return historical_data