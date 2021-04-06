from RLAgent import RLAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
np.set_printoptions(threshold=np.inf)

import math
import sumolib
import sys

ACTION_SPACE = 2
STATE_SPACE = (25, 25)

MIN_GREEN_VEHICLE = 20
MAX_RED_VEHICLE = 30

ARRAY_LENGTH = 9
CENTER_LENGTH = 1
MAP_SIZE = (2*(ARRAY_LENGTH + CENTER_LENGTH), 2*(ARRAY_LENGTH + CENTER_LENGTH))

LENGTH_CELL = 5

# we only support 3-way and 4-way intersections
MAX_NUM_WAY = 4
# we assume that there are only 2 red/green phases, user can change this depend on their config
NUM_OF_RED_GREEN_PHASES = 2

class CDRL(RLAgent):
    def __init__(self, config=None, tfID=None):
        RLAgent.__init__(self)
        self.config = config
        self.tfID = tfID
        
        # self.buildMap()

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
        neightbor_nodes_sorted = [neightbor_nodes[1], neightbor_nodes[0], neightbor_nodes[2], neightbor_nodes[3]]
        # center_node_coord = center_node.getCoord()
        # nodes_id = [node.getID() for node in neightbor_nodes_sorted]

        return neightbor_nodes_sorted, center_node

    def processState(self, state=None):
        """
            from general state returned from traffic light
            process to return (current_logic, num_veh_ordered)
            current_logic: 'ggggrrrrgggg' shows status of traffic light
            num_veh_ordered: [1, 2, 1, 5, ...] shows number of vehicles on each lane by order  
        """
        current_logic = state['current_logic']
        num_veh_ordered = []

        for lane in state['lanes']:
            num_veh_ordered.append(state['traci'].lane.getLastStepVehicleNumber(lane))

        # print(state['lanes'])
        # print(np.array_str(self.build_map(state['traci']), precision=2, suppress_small=True))
        map_ = self.buildMap(traci=state['traci'])
        print(np.array_str(map_, suppress_small=True))
        print("")
        return current_logic, num_veh_ordered


    def buildMap(self, traci=None):
        """
            this function to return a 2D matrix indicating information on vehicles' positions
        """
        # ['NtoC_0', 'NtoC_1', 'EtoC_0', 'EtoC_1', 'StoC_0', 'StoC_1', 'WtoC_0', 'WtoC_1']
        neightbor_nodes, center_node = self.getNodesSortedByDirection()
        incoming_edges, outgoing_edges = center_node.getIncoming(), center_node.getOutgoing()

        

        position_mapped = np.zeros(MAP_SIZE)

        # handle the North side
        if neightbor_nodes[0] != None:
            incoming_edge_from_north = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[0].getID()][0]
            outgoing_edge_to_north = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[0].getID()][0]
            for i, lane in enumerate(incoming_edge_from_north.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=True)
                for j in range(ARRAY_LENGTH):
                    position_mapped[j][ARRAY_LENGTH + CENTER_LENGTH + i - incoming_edge_from_north.getLaneNumber()] = arr_[j]
            for i, lane in enumerate(outgoing_edge_to_north.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=False)[::-1]
                for j in range(ARRAY_LENGTH):
                    position_mapped[j][ARRAY_LENGTH + CENTER_LENGTH + outgoing_edge_to_north.getLaneNumber() - i - 1] = arr_[j]
        

        # handle the East side
        if neightbor_nodes[1] != None:
            incoming_edge_from_east = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[1].getID()][0]
            outgoing_edge_to_east = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[1].getID()][0]
            for i, lane in enumerate(incoming_edge_from_east.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=True)[::-1]
                for j in range(ARRAY_LENGTH):
                    position_mapped[ARRAY_LENGTH + CENTER_LENGTH - incoming_edge_from_east.getLaneNumber() + i][ARRAY_LENGTH + CENTER_LENGTH + j] = arr_[j]
            for i, lane in enumerate(outgoing_edge_to_east.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=False)
                for j in range(ARRAY_LENGTH):
                    position_mapped[ARRAY_LENGTH + CENTER_LENGTH + outgoing_edge_to_east.getLaneNumber() - i - 1][ARRAY_LENGTH + CENTER_LENGTH + j] = arr_[j]

        # handle the South side
        if neightbor_nodes[2] != None:
            incoming_edge_from_south = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[2].getID()][0]
            outgoing_edge_to_south = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[2].getID()][0]
            for i, lane in enumerate(incoming_edge_from_south.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=True)[::-1]
                for j in range(ARRAY_LENGTH):
                    position_mapped[j + ARRAY_LENGTH + CENTER_LENGTH][ARRAY_LENGTH + CENTER_LENGTH + incoming_edge_from_south.getLaneNumber() - i - 1] = arr_[j]

            for i, lane in enumerate(outgoing_edge_to_south.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=False)
                for j in range(ARRAY_LENGTH):
                    position_mapped[j + ARRAY_LENGTH + CENTER_LENGTH][ARRAY_LENGTH + CENTER_LENGTH - outgoing_edge_to_south.getLaneNumber() + i] = arr_[j]

        # handle the west side
        if neightbor_nodes[3] != None:
            incoming_edge_from_west = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[3].getID()][0]
            outgoing_edge_to_west = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[3].getID()][0]
            for i, lane in enumerate(incoming_edge_from_west.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=True)
                for j in range(ARRAY_LENGTH):
                    position_mapped[ARRAY_LENGTH + CENTER_LENGTH + outgoing_edge_to_west.getLaneNumber() - i - 1][j] = arr_[j]
            for i, lane in enumerate(outgoing_edge_to_west.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=False)[::-1]
                for j in range(ARRAY_LENGTH):
                    position_mapped[ARRAY_LENGTH + CENTER_LENGTH - incoming_edge_from_west.getLaneNumber() + i][j] = arr_[j]

        return position_mapped
    
    def buildArray(self, traci=None, lane=None, incoming=True):
        arr = np.zeros(ARRAY_LENGTH)
        # lane = 'CtoW_0', 'EtoC_0' It is inverted for this case
        lane_length = traci.lane.getLength(lane)
        vehs = traci.lane.getLastStepVehicleIDs(lane)
        for veh in vehs:
            veh_distance = traci.vehicle.getLanePosition(veh)

            if incoming:
                veh_distance -= lane_length - LENGTH_CELL*ARRAY_LENGTH
            if veh_distance < 0:
                continue
            index = math.floor(veh_distance/5)

            if index >= ARRAY_LENGTH:
                continue
            veh_length = traci.vehicle.getLength(veh)
            for i in range(math.ceil(veh_length/5)):
                if 0 <= index - i < ARRAY_LENGTH:
                    arr[index - i] = 1

        return arr

    def makeAction(self, state_):
        """
            return action based on SOTL's rules & current state
        """
        state = self.processState(state_)
        current_logic, num_veh_ordered = state
        number_veh_on_green_lanes = 0
        number_veh_on_red_lanes = 0

        for i in range(len(num_veh_ordered)):
            if current_logic[i] in ['r', 'R']:
                number_veh_on_red_lanes += num_veh_ordered[i]
            elif current_logic[i] in ['g', 'G']:
                number_veh_on_green_lanes += num_veh_ordered[i]
            else:
                print(state, "Error")
        if (number_veh_on_green_lanes < MIN_GREEN_VEHICLE and number_veh_on_red_lanes > MAX_RED_VEHICLE) or (number_veh_on_green_lanes == 0 and number_veh_on_red_lanes > 0):
            return 1
        return 0