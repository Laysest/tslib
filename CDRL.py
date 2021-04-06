from RLAgent import RLAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
import math
import sumolib

ACTION_SPACE = 2
STATE_SPACE = (25, 25)

MIN_GREEN_VEHICLE = 20
MAX_RED_VEHICLE = 30

ARRAY_LENGTH = 20
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
        neightbor_nodes_sorted = [neightbor_nodes[2], neightbor_nodes[1], neightbor_nodes[3], neightbor_nodes[0]]

        # center_node_coord = center_node.getCoord()
        # nodes_id = [node.getID() for node in neightbor_nodes_sorted]

        return neightbor_nodes_sorted, center_node


    

    # def processState(self, state):
    #     """
    #         from general state returned from traffic light
    #         process to return (number_veh_on_green_lanes, number_veh_on_red_lanes)
    #     """

    #     num_veh_ordered = []
    #     for lane in state['lanes']:
    #         num_veh_ordered.append(state['traci'].lane.getLastStepVehicleNumber(lane))

    #     number_veh_on_green_lanes = 0
    #     number_veh_on_red_lanes = 0
    #     for i in range(len(num_veh_ordered)):
    #         if state['current_logic'][i] in ['r', 'R']:
    #             number_veh_on_red_lanes += num_veh_ordered[i]
    #         elif state['current_logic'][i] in ['g', 'G']:
    #             number_veh_on_green_lanes += num_veh_ordered[i]
    #         else:
    #             print("Error in getState in case of SimpleRL")

    #     return [number_veh_on_green_lanes, number_veh_on_red_lanes]
    
    # def computeReward(self, state):
    #     reward = 0
    #     for lane in state['lanes']:
    #         reward -= state['traci'].lane.getLastStepHaltingNumber(lane)
    #     return reward

    # def buildModel(self):
    #     """
    #         return the model in keras
    #     """
    #     model = Sequential()
    #     model.add(Dense(16, input_dim=STATE_SPACE))
    #     model.add(Activation('relu'))
    #     model.add(Dense(32))
    #     model.add(Activation('relu'))
    #     model.add(Dense(32))
    #     model.add(Activation('relu'))
    #     model.add(Dense(ACTION_SPACE))
    #     model.add(Activation('linear'))
    #     model.compile(loss='mean_squared_error', optimizer='adam')

    #     return model

    def processState(self, state):
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
        self.buildMap(state['traci'], state['lanes'])

        return current_logic, num_veh_ordered


    def buildMap(self, traci, lanes):
        """
            this function to return a 2D matrix indicating information on vehicles' positions
        """
        # ['NtoC_0', 'NtoC_1', 'EtoC_0', 'EtoC_1', 'StoC_0', 'StoC_1', 'WtoC_0', 'WtoC_1']
        
        # for lane in lanes:
        #     print(traci.lane.getLastStepVehicleIDs(lane))
        arr = self.build_array(traci, lanes[0])
        # for a in arr:
        #     print(a)
        # print('')
        print(arr)

        return np.array([[1, 1, 1], [0.99999, 0.954, 0.5124], [1, 1, 1]])
    
    def buildArray(self, traci, lane):
        arr = np.zeros(ARRAY_LENGTH)
        lane = 'StoC_0'
        # lane = 'CtoW_0', 'EtoC_0' It is inverted for this case
        lane_length = traci.lane.getLength(lane)
        vehs = traci.lane.getLastStepVehicleIDs(lane)
        for veh in vehs:
            veh_distance = traci.vehicle.getLanePosition(veh)

            veh_distance -= lane_length - LENGTH_CELL*ARRAY_LENGTH
            if veh_distance < 0:
                continue
            index = math.floor(veh_distance/5)

            if index >= ARRAY_LENGTH:
                continue
            veh_length = traci.vehicle.getLength(veh)
            for i in range(math.ceil(veh_length/5)):
                if index + i < ARRAY_LENGTH:
                    arr[index + i] = 1

        return arr

    def buildMap(self):
        center_node, neightbor_nodes = self.getNodesSortedByDirection()
        position_mapped = np.zeros((ARRAY_LENGTH, ARRAY_LENGTH))


        return position_mapped

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