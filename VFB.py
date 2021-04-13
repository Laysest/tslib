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
from GloVars import GloVars

traci = GloVars.traci
# we only support 3-way and 4-way intersections
MAX_NUM_WAY = 4
# we assume that there are only 2 red/green phases, user can change this depend on their config
NUM_OF_RED_GREEN_PHASES = 2

class VFB(RLAgent):
    def __init__(self, config=None, tfID=None):
        RLAgent.__init__(self)
        self.config = config
        self.tfID = tfID
        nodes, center = self.getNodesSortedByDirection()
        nodes_id = [node.getID() for node in nodes]
        print("%s: %s" % (center.getID(), str(nodes_id)))

    def computeReward(self, state, last_state):
        reward = 0

        # get list vehicles
        lanes = list(dict.fromkeys(state['lanes']))
        vehs = []
        for lane in lanes:
            vehs.extend(traci.lane.getLastStepVehicleIDs(lane))
        
        # total delay
        total_delay = 0
        for veh in vehs:
            total_delay += 1 - traci.vehicle.getSpeed(veh) / traci.vehicle.getAllowedSpeed(veh)
        
        reward = state['last_total_delay'] - total_delay

        return reward

    def buildModel(self):
        """
            return the model in keras
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=GloVars.STATE_SPACE))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(GloVars.ACTION_SPACE, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def getNodesSortedByDirection(self):
        """
            This function will return a list of nodes sorted by direction
                N
            W       E
                S
            [N, E, S, W] for 4-way intersection

                N
            W       E
            [N, E, None, W]

        """
        
        center_node = sumolib.net.readNet('./traffic-sumo/%s' % self.config['net']).getNode(self.tfID)
        neightbor_nodes = center_node.getNeighboringNodes()
        # isolated...
        # neightbor_nodes_sorted = [neightbor_nodes[1], neightbor_nodes[0], neightbor_nodes[2], neightbor_nodes[3]]
        # 4x1 network
        neightbor_nodes_sorted = [neightbor_nodes[2], neightbor_nodes[1], neightbor_nodes[3], neightbor_nodes[0]]
        
        # center_node_coord = center_node.getCoord()
        return neightbor_nodes_sorted, center_node

    def processState(self, state=None):
        """
            from general state returned from traffic light
            process to return position_map
        """
        return np.reshape(self.buildMap(), GloVars.STATE_SPACE) # reshape to (SPACE, 1)


    def buildMap(self):
        """
            this function to return a 2D matrix indicating information on vehicles' positions
        """
        # ['NtoC_0', 'NtoC_1', 'EtoC_0', 'EtoC_1', 'StoC_0', 'StoC_1', 'WtoC_0', 'WtoC_1']
        neightbor_nodes, center_node = self.getNodesSortedByDirection()
        incoming_edges, outgoing_edges = center_node.getIncoming(), center_node.getOutgoing()

        

        position_mapped = np.zeros(GloVars.MAP_SIZE)

        # handle the North side
        if neightbor_nodes[0] != None:
            incoming_edge_from_north = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[0].getID()][0]
            outgoing_edge_to_north = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[0].getID()][0]
            for i, lane in enumerate(incoming_edge_from_north.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=True)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + i - incoming_edge_from_north.getLaneNumber()] = arr_[j]
            for i, lane in enumerate(outgoing_edge_to_north.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=False)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + outgoing_edge_to_north.getLaneNumber() - i - 1] = arr_[j]
        

        # handle the East side
        if neightbor_nodes[1] != None:
            incoming_edge_from_east = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[1].getID()][0]
            outgoing_edge_to_east = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[1].getID()][0]
            for i, lane in enumerate(incoming_edge_from_east.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=True)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH - incoming_edge_from_east.getLaneNumber() + i][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + j + 1] = arr_[j]
            for i, lane in enumerate(outgoing_edge_to_east.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=False)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + outgoing_edge_to_east.getLaneNumber() - i - 1][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + j + 1] = arr_[j]

        # handle the South side
        if neightbor_nodes[2] != None:
            incoming_edge_from_south = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[2].getID()][0]
            outgoing_edge_to_south = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[2].getID()][0]
            for i, lane in enumerate(incoming_edge_from_south.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=True)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j + GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + 1][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + incoming_edge_from_south.getLaneNumber() - i - 1] = arr_[j]

            for i, lane in enumerate(outgoing_edge_to_south.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=False)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[j + GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + 1][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH - outgoing_edge_to_south.getLaneNumber() + i] = arr_[j]

        # handle the West side
        if neightbor_nodes[3] != None:
            incoming_edge_from_west = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[3].getID()][0]
            outgoing_edge_to_west = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[3].getID()][0]
            for i, lane in enumerate(incoming_edge_from_west.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=True)
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH + outgoing_edge_to_west.getLaneNumber() - i - 1][j] = arr_[j]
            for i, lane in enumerate(outgoing_edge_to_west.getLanes()):
                arr_ = self.buildArray(lane=lane.getID(), incoming=False)[::-1]
                for j in range(GloVars.ARRAY_LENGTH):
                    position_mapped[GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH - incoming_edge_from_west.getLaneNumber() + i][j] = arr_[j]

        return position_mapped
    
    def buildArray(self, lane=None, incoming=True):
        arr = np.zeros(GloVars.ARRAY_LENGTH)
        # lane = 'CtoW_0', 'EtoC_0' It is inverted for this case
        lane_length = traci.lane.getLength(lane)
        vehs = traci.lane.getLastStepVehicleIDs(lane)
        for veh in vehs:
            veh_distance = traci.vehicle.getLanePosition(veh)

            if incoming:
                veh_distance -= lane_length - GloVars.LENGTH_CELL*GloVars.ARRAY_LENGTH
            if veh_distance < 0:
                continue
            index = math.floor(veh_distance/5)

            if index >= GloVars.ARRAY_LENGTH:
                continue
            veh_length = traci.vehicle.getLength(veh)
            for i in range(math.ceil(veh_length/5)):
                if 0 <= index - i < GloVars.ARRAY_LENGTH:
                    arr[index - i] = 1

        return arr