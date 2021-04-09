from RLAgent import RLAgent
from controller import ActionType
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
np.set_printoptions(threshold=np.inf)

import math
import sumolib
import sys

ACTION_SPACE = 2

MIN_GREEN_VEHICLE = 20
MAX_RED_VEHICLE = 30

ARRAY_LENGTH = 9
CENTER_LENGTH = 1
MAP_SIZE = (2*(ARRAY_LENGTH + CENTER_LENGTH), 2*(ARRAY_LENGTH + CENTER_LENGTH))
STATE_SPACE = (MAP_SIZE[0], MAP_SIZE[1], 1)

LENGTH_CELL = 5

# we only support 3-way and 4-way intersections
MAX_NUM_WAY = 4
# we assume that there are only 2 red/green phases, user can change this depend on their config
NUM_OF_RED_GREEN_PHASES = 2

BATCH_SIZE = 64
GAMMA = 0.95
EPOCHS = 50
SAMPLE_SIZE = 256

class IntelliLight(RLAgent):
    def __init__(self, config=None, tfID=None):
        RLAgent.__init__(self)
        self.config = config
        self.tfID = tfID
        nodes, center = self.getNodesSortedByDirection()
        nodes_id = [node.getID() for node in nodes]
        print("%s: %s" % (center.getID(), str(nodes_id)))

    def computeReward(self, state):
        reward = 0

        # penalty for change signal
        reward -= 0.1*state['last_action_is_change']
        
        # get list vehicles
        lanes = list(dict.fromkeys(state['lanes']))
        vehs = []
        for lane in lanes:
            vehs.extend(state['traci'].lane.getLastStepVehicleIDs(lane))
        
        # penalty for teleports
        num_veh_teleporting = 0
        vehs_teleporting = state['traci'].simulation.getStartingTeleportIDList()
        for veh in vehs:
            if veh in vehs_teleporting:
                num_veh_teleporting += 1
        reward -= 0.1*num_veh_teleporting
        
        # penalty for emergency stops
        num_veh_emergency_stop = 0
        vehs_emergency_stop = state['traci'].simulation.getEmergencyStoppingVehiclesIDList()
        for veh in vehs:
            if veh in vehs_emergency_stop:
                num_veh_emergency_stop += 1
        reward -= 0.2*num_veh_emergency_stop

        # penalty for delay
        total_delay = 0
        for veh in vehs:
            total_delay += 1 - state['traci'].vehicle.getSpeed(veh) / state['traci'].vehicle.getAllowedSpeed(veh)
        reward -= 0.3*total_delay

        # penalty for waiting time
        total_waiting_time = 0
        for veh in vehs:
            total_waiting_time += state['traci'].vehicle.getWaitingTime(veh)
        reward -= 0.3*total_waiting_time

        return reward

    def buildModel(self):
        """
            return the model in keras
        """
        # model = Sequential()
        input_ = Input(shape=STATE_SPACE)
        phase_ = Input(shape=(1))
        
        image_1_ = Conv2D(32, (3, 3), activation='relu')(input_)
        polling_1_ = MaxPooling2D((2, 2))(image_1_)
        flatten_1_ = Flatten()(polling_1_)
        dense_1_1_ = Dense(128, activation='relu')(flatten_1_)
        dense_2_1_ = Dense(32, activation='relu')(dense_1_1_)
        out_1_ = Dense(ACTION_SPACE, activation='linear')(dense_2_1_)

        image_2_ = Conv2D(32, (3, 3), activation='relu')(input_)
        polling_2_ = MaxPooling2D((2, 2))(image_2_)
        flatten_2_ = Flatten()(polling_2_)
        dense_1_2_ = Dense(128, activation='relu')(flatten_2_)
        dense_2_2_ = Dense(32, activation='relu')(dense_1_2_)
        out_2_ = Dense(ACTION_SPACE, activation='linear')(dense_2_2_)

        out = tf.where(phase_ == 0, out_1_, out_2_)

        model = tf.keras.Model(inputs=[input_, phase_], outputs=out)
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
        map_ = [np.reshape(self.buildMap(traci=state['traci']), STATE_SPACE)] # reshape to (SPACE, 1)
        phase = state['traci'].trafficlight.getPhase(self.tfID)
        state_ = [np.array(map_), np.array([phase])]
        return state_

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
                    position_mapped[ARRAY_LENGTH + CENTER_LENGTH - incoming_edge_from_east.getLaneNumber() + i][ARRAY_LENGTH + CENTER_LENGTH + j + 1] = arr_[j]
            for i, lane in enumerate(outgoing_edge_to_east.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=False)
                for j in range(ARRAY_LENGTH):
                    position_mapped[ARRAY_LENGTH + CENTER_LENGTH + outgoing_edge_to_east.getLaneNumber() - i - 1][ARRAY_LENGTH + CENTER_LENGTH + j + 1] = arr_[j]

        # handle the South side
        if neightbor_nodes[2] != None:
            incoming_edge_from_south = [edge for edge in incoming_edges if edge.getFromNode().getID() == neightbor_nodes[2].getID()][0]
            outgoing_edge_to_south = [edge for edge in outgoing_edges if edge.getToNode().getID() == neightbor_nodes[2].getID()][0]
            for i, lane in enumerate(incoming_edge_from_south.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=True)[::-1]
                for j in range(ARRAY_LENGTH):
                    position_mapped[j + ARRAY_LENGTH + CENTER_LENGTH + 1][ARRAY_LENGTH + CENTER_LENGTH + incoming_edge_from_south.getLaneNumber() - i - 1] = arr_[j]

            for i, lane in enumerate(outgoing_edge_to_south.getLanes()):
                arr_ = self.buildArray(traci=traci, lane=lane.getID(), incoming=False)
                for j in range(ARRAY_LENGTH):
                    position_mapped[j + ARRAY_LENGTH + CENTER_LENGTH + 1][ARRAY_LENGTH + CENTER_LENGTH - outgoing_edge_to_south.getLaneNumber() + i] = arr_[j]

        # handle the West side
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

        return self.addSignalInfor(position_mapped, traci.trafficlight.getPhase(self.tfID))
    
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

    def addSignalInfor(self, position_mapped, cur_phase):
        neightbor_nodes, center_node = self.getNodesSortedByDirection()

        # 4-way intersection
        if None not in neightbor_nodes:
            # cur_phase == 0 ~ allow N and S
            if cur_phase == 0:
                position_mapped[ARRAY_LENGTH][ARRAY_LENGTH], position_mapped[ARRAY_LENGTH+CENTER_LENGTH][ARRAY_LENGTH+CENTER_LENGTH] = 0.8, 0.8
                position_mapped[ARRAY_LENGTH][ARRAY_LENGTH + CENTER_LENGTH], position_mapped[ARRAY_LENGTH+CENTER_LENGTH][ARRAY_LENGTH] = 0.2, 0.2
            elif cur_phase == 2:
                position_mapped[ARRAY_LENGTH][ARRAY_LENGTH], position_mapped[ARRAY_LENGTH+CENTER_LENGTH][ARRAY_LENGTH+CENTER_LENGTH] = 0.2, 0.2
                position_mapped[ARRAY_LENGTH][ARRAY_LENGTH + CENTER_LENGTH], position_mapped[ARRAY_LENGTH+CENTER_LENGTH][ARRAY_LENGTH] = 0.8, 0.8
            else:
                print("Error in CRDL.py - addSignalInfor()")
        # 3-way intersection
        else:
            pass

        return position_mapped
    
    def actionType(self):
        return ActionType.CHANGING_KEEPING

    def replay(self):
        if self.exp_memory.len() < SAMPLE_SIZE:
            return
        minibatch =  self.exp_memory.sample(SAMPLE_SIZE)    
        batch_states = []
        batch_targets = []
        for state_, action_, reward_, next_state_ in minibatch:
            qs = self.model.predict([next_state_])
            target = reward_ + GAMMA*np.amax(qs[0])
            target_f = self.model.predict([state_])
            target_f[0][action_] = target
            batch_states.append(state_)
            batch_targets.append(target_f[0])
        # for i in range(len(batch_states)):
        #     print(np.array(batch_states[i][0]).shape, np.array(batch_states[i][1]).shape)
        # self.model.train_on_batch(batch_states, batch_targets)
        print("replay")
        for epoch in range(EPOCHS):
            self.model.train_on_batch(batch_states, batch_targets)
            # self.model.train_on_batch(batch_states, batch_targets)

    def makeAction(self, state):
        state_ = self.processState(state)
        # print(np.array(state_[0]).shape, np.array(state_[1]).shape) 
        out_ = self.model.predict([state_])[0]
        return np.argmax(out_)