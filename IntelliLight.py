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


traci = GloVars.traci

# FEATURE_SPACE = (8*4)
PHASE_SPACE = (1)

# we only support 3-way and 4-way intersections
MAX_NUM_WAY = 4
# we assume that there are only 2 red/green phases, user can change this depend on their config
NUM_OF_RED_GREEN_PHASES = 2

class IntelliLight(RLAgent):
    def __init__(self, config=None, tf_id=None):        
        self.config = config
        self.tf_id = tf_id
        nodes, center = self.getNodesSortedByDirection()
        nodes_id = [node.getID() for node in nodes]
        self.lanes = traci.trafficlight.getControlledLanes(self.tf_id)
        self.lanes_unique = list(dict.fromkeys(self.lanes))
        RLAgent.__init__(self, config['cycle_control'])

        print("%s: %s" % (center.getID(), str(nodes_id)))

    def computeReward(self, state, historical_data):
        reward = 0
        # penalty for queue lengths:
        L = 0
        for lane in self.lanes_unique:
            L += traci.lane.getLastStepHaltingNumber(lane)
        reward -= 0.25 * L

        # penalty for delay
        D = 0
        for lane in self.lanes_unique:
            D += 1 - traci.lane.getLastStepMeanSpeed(lane) / traci.lane.getMaxSpeed(lane)
        reward -= 0.25*D

        # penalty for waiting time
        W = 0
        for lane in self.lanes_unique:
            W += traci.lane.getWaitingTime(lane) / 60.0
        reward -= 0.25*W

        # penalty for change
        reward -= 5*historical_data['IntelliLight']['last_action_is_change']

        # reward for number vehicles
        N = 0
        vehs = []
        for lane in self.lanes_unique:
            vehs.extend(traci.lane.getLastStepVehicleIDs(lane))
        vehs_id_passed = []
        for veh_id_ in historical_data['IntelliLight']['last_vehs_id']:
            # if a veh in vehs_id but not in current vehs => passed
            if veh_id_ not in vehs:
                N += 1
                vehs_id_passed.append(veh_id_)
        reward += N

        #reward for travel time of passed vehicles
        total_travel_time = 0 
        for veh_id_ in vehs_id_passed:
            if (veh_id_ in GloVars.vehicles.keys()) and ((traci.simulation.getTime() - 1) in GloVars.vehicles[veh_id_].log.keys()):
                veh_route_ = GloVars.vehicles[veh_id_].log[traci.simulation.getTime() - 1]['route']
                if len(veh_route_) >= 3:
                    total_travel_time += veh_route_[-3]['last_time_step'] - veh_route_[-3]['first_time_step']
        reward += total_travel_time/60

        return reward

    def buildModel(self):
        """
            return the model in keras
        """
        # model = Sequential()
        map_ = Input(shape=GloVars.STATE_SPACE)
        lane_features_ = Input(shape=4*len(self.lanes_unique))
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
        center_node = sumolib.net.readNet('./traffic-sumo/%s' % GloVars.config['net']).getNode(self.tf_id)
        neightbor_nodes = center_node.getNeighboringNodes()
        # isolated...
        # neightbor_nodes_sorted = [neightbor_nodes[1], neightbor_nodes[0], neightbor_nodes[2], neightbor_nodes[3]]
        # 4x1 network
        # neightbor_nodes_sorted = [neightbor_nodes[2], neightbor_nodes[1], neightbor_nodes[3], neightbor_nodes[0]]
        
        # center_node_coord = center_node.getCoord()
        return neightbor_nodes, center_node

    def processState(self, state=None):
        """
            from general state returned from traffic light
            process to return position_map
        """
        map_ = np.reshape(self.buildMap(), GloVars.STATE_SPACE) # reshape to (SPACE, 1)
        
        lane_features_ = self.getLaneFeatures()

        phase = traci.trafficlight.getPhase(self.tf_id)
        state_ = [np.array(map_), np.array(lane_features_), np.array([phase])]
        return state_

    def getLaneFeatures(self):
        all_logic_ = traci.trafficlight.getAllProgramLogics(self.tf_id)[0]
        current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
        # lanes = list(dict.fromkeys(lanes_in_phases))
        lane_features_ = []

        # queue length
        for lane in self.lanes_unique:
            lane_features_.append(traci.lane.getLastStepHaltingNumber(lane))
        
        # waiting time
        for lane in self.lanes_unique:
            lane_features_.append(traci.lane.getWaitingTime(lane))

        # phase vector
        for lane in self.lanes_unique:
            lane_features_.append(1 if current_logic[self.lanes.index(lane)].lower() == 'g' else 0)

        # number of vehicles
        for lane in self.lanes_unique:
            lane_features_.append(traci.lane.getLastStepVehicleNumber(lane))

        return lane_features_

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
        if len(neightbor_nodes) > 3 and neightbor_nodes[3] != None:
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

        return self.addSignalInfor(position_mapped, traci.trafficlight.getPhase(self.tf_id))
    
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

    def addSignalInfor(self, position_mapped, cur_phase):
        neightbor_nodes, center_node = self.getNodesSortedByDirection()

        # 4-way intersection
        if None not in neightbor_nodes:
            # cur_phase == 0 ~ allow N and S
            if cur_phase == 0:
                position_mapped[GloVars.ARRAY_LENGTH][GloVars.ARRAY_LENGTH], position_mapped[GloVars.ARRAY_LENGTH+GloVars.CENTER_LENGTH][GloVars.ARRAY_LENGTH+GloVars.CENTER_LENGTH] = 0.8, 0.8
                position_mapped[GloVars.ARRAY_LENGTH][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH], position_mapped[GloVars.ARRAY_LENGTH+GloVars.CENTER_LENGTH][GloVars.ARRAY_LENGTH] = 0.2, 0.2
            elif cur_phase == 2:
                position_mapped[GloVars.ARRAY_LENGTH][GloVars.ARRAY_LENGTH], position_mapped[GloVars.ARRAY_LENGTH+GloVars.CENTER_LENGTH][GloVars.ARRAY_LENGTH+GloVars.CENTER_LENGTH] = 0.2, 0.2
                position_mapped[GloVars.ARRAY_LENGTH][GloVars.ARRAY_LENGTH + GloVars.CENTER_LENGTH], position_mapped[GloVars.ARRAY_LENGTH+GloVars.CENTER_LENGTH][GloVars.ARRAY_LENGTH] = 0.8, 0.8
            else:
                print("Error in CRDL.py - addSignalInfor()")
        # 3-way intersection
        else:
            pass

        return position_mapped
    
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

    