"""
    This file declare TrafficLight class
"""
import enum
import random
import sys
import os
from numpy.core.fromnumeric import sort
import sumolib
import json
import tensorflow as tf
import pandas as pd
import math
from SOTL import SOTL
from CDRL import CDRL
from VFB import VFB
from IntelliLight import IntelliLight
from LIT import LIT
from LITP import LITP
from TLCC import TLCC
from FixedTime import FixedTime
from MaxPressure import MaxPressure
from CAREL import CAREL
from glo_vars import GloVars
from controller import ActionType, Controller


MAX_INT = 9999999
traci = GloVars.traci

# TODO: change it to more general
MAX_NUM_PHASE = 4

MAX_WAY_OF_AN_INTERSECTION = 4

class LightState:
    Green = 0
    Yellow = 1
    Red = 2

class TrafficLight:
    # pylint: disable=line-too-long invalid-name too-many-instance-attributes
    '''
        TrafficLight for each intersection having traffic signal control
    '''
    def __init__(self, config=None):
        self.id = config['node_id']
        self.control_algorithm = config['method']
        self.yellow_duration = config['yellow_duration']
        self.cycle_control = config['cycle_control']
        self.folder = config['folder']
        self.number_of_phases, self.phase_description = self.getPhase()
        if self.number_of_phases % 2 != 0:
            print("<<< To ensure safety, must have a yellow phase after each changing phase >>>")
            sys.exit(0)
        self.road_structure = self.getRoadStructure()
        self.lanes_id = []
        for road in self.road_structure.keys():
            self.lanes_id.extend([lane['id'] for lane in self.road_structure[road]])

        if self.control_algorithm == 'SOTL':
            self.controller = SOTL(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'CDRL':
            self.controller = CDRL(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'VFB':
            self.controller = VFB(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'IntelliLight':
            self.controller = IntelliLight(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'LIT':
            self.controller = LIT(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'LITP':
            self.controller = LITP(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'TLCC':
            self.controller = TLCC(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'FixedTime':
            self.controller = FixedTime(config=config, tfl_id=self.id)
        elif self.control_algorithm == 'MaxPressure':
            self.controller = MaxPressure(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)
        elif self.control_algorithm == 'CAREL':
            self.controller = CAREL(config=config, road_structure=self.road_structure, number_of_phases=self.number_of_phases)

        else:
            print("<<< Must implement %s >>>" % self.control_algorithm)
            sys.exit(0)

        self.writer = tf.summary.create_file_writer('%s/%s-%s' % (self.folder, self.id, self.control_algorithm))
        self.last_action, self.last_processed_state, self.last_state, self.historical_data = None, None, None, None
        self.current_phase_index = 0
        self.reset()
        

    def reset(self):
        if self.control_algorithm != 'FixedTime':
            self.setLogic()
        self.control_actions = []
        self.last_action, self.last_processed_state, self.last_state, self.historical_data = None, None, None, None
        self.current_phase_index = 0

    def getPhase(self):
        def getPhaseDescription(tfl_id):
            if GloVars.config['simulator'] == 'SUMO':
                phases = traci.trafficlight.getAllProgramLogics(tfl_id)[0].getPhases()
                links = traci.trafficlight.getControlledLinks(tfl_id)
                def map_phase_state(state):
                    if state in ['r', 'R']:
                        return LightState.Red
                    if state in ['g', 'G']:
                        return LightState.Green
                    return LightState.Yellow
                des = []
                for idx, phase in enumerate(phases):
                    tmp = []
                    for link in links:
                        tmp.append({
                            'from': link[0][0],
                            'to': link[0][1],
                            'light_state': map_phase_state(phase.state[idx])
                        })
                    des.append(tmp)
                return des
            elif GloVars.config['simulator'] == 'CityFlow':
                config_CF = json.load(open(GloVars.config['config_file'], 'r'))
                road_structure = json.load(open(config_CF['dir'] + config_CF['roadnetFile'], 'r'))
                intersection = next(item for item in road_structure['intersections'] if item["id"] == tfl_id)
                phases = intersection['trafficLight']['lightphases']
                road_links = intersection['roadLinks']

                des = []
                for phase in phases:
                    tmp = []
                    for link_idx, link in enumerate(road_links):
                        for lane_link in link['laneLinks']:
                            if link_idx in phase['availableRoadLinks']:
                                tmp.append({
                                    'from': "%s_%d" % (link['startRoad'], lane_link['startLaneIndex']),
                                    'to': "%s_%d" % (link['endRoad'], lane_link['endLaneIndex']),
                                    'light_state': LightState.Green
                                })
                            else:
                                tmp.append({
                                    'from': "%s_%d" % (link['startRoad'], lane_link['startLaneIndex']),
                                    'to': "%s_%d" % (link['endRoad'], lane_link['endLaneIndex']),
                                    'light_state': LightState.Red
                                })
                    des.append(tmp)
                return des
                
        def getNumberOfPhases(tfl_id):
            if GloVars.config['simulator'] == 'SUMO':
                return len(traci.trafficlight.getAllProgramLogics(tfl_id)[0].getPhases())            
            elif GloVars.config['simulator'] == 'CityFlow':
                config_CF = json.load(open(GloVars.config['config_file'], 'r'))
                road_structure = json.load(open(config_CF['dir'] + config_CF['roadnetFile'], 'r'))
                intersection = next(item for item in road_structure['intersections'] if item["id"] == tfl_id)
                return len(intersection['trafficLight']['lightphases'])

        return getNumberOfPhases(self.id), getPhaseDescription(self.id)

    def setLogic(self):
        """
            ### To do ###
            Set logic program for the traffic light
            Restart the logic at phase 0
        """
        if GloVars.config['simulator'] == 'SUMO':
            traci.trafficlight.setPhase(self.id, 0)
            traci.trafficlight.setPhaseDuration(self.id, MAX_INT)
        
    def getRoadStructure(self):
        class Node:
            node_id = None
            x = None   
            y = None

        def getNodes(tfl_id):
            if GloVars.config['simulator'] == 'SUMO':
                center = sumolib.net.readNet('./traffic-sumo/%s' % GloVars.config['net']).getNode(tfl_id)
                center_node = Node()
                center_node.node_id, (center_node.x, center_node.y) = tfl_id, center.getCoord()
                incoming = center.getNeighboringNodes(incomingNodes=True)
                incoming_nodes = []
                for item in incoming:
                    tmp = Node()
                    tmp.node_id, (tmp.x, tmp.y) = item.getID(), item.getCoord()
                    incoming_nodes.append(tmp)
            elif GloVars.config['simulator'] == 'CityFlow':
                config_CF = json.load(open(GloVars.config['config_file'], 'r'))
                road_structure = json.load(open(config_CF['dir'] + config_CF['roadnetFile'], 'r'))
                intersection = next(item for item in road_structure['intersections'] if item["id"] == tfl_id)
                center_node = Node()
                center_node.node_id, center_node.x, center_node.y = intersection['id'], intersection['point']['x'], intersection['point']['y']
                
                start_roads_id = []
                for link in intersection['roadLinks']:
                    if link['startRoad'] not in start_roads_id:
                        start_roads_id.append(link['startRoad'])
                incoming_nodes = []
                for road_id in start_roads_id:
                    for road in road_structure['roads']:
                        if road['id'] == road_id:
                            available_nodes_id = [node.node_id for node in incoming_nodes]
                            if road['startIntersection'] in available_nodes_id:
                                break
                            tmp = Node()
                            tmp.node_id, tmp.x, tmp.y = road['startIntersection'], road['points'][0]['x'], road['points'][0]['y']
                            incoming_nodes.append(tmp)
                            break
            return center_node, incoming_nodes
        
        center_node, incoming_nodes = getNodes(self.id)
        sorted_nodes = []
        if len(incoming_nodes) > 4:
            print("<<<Currently TSlib supports only maximum 4-way intersection>>>")
            sys.exit(0)

        if len(incoming_nodes) == 4:
            # find West node
            min_distance, selected_node = -1, None
            for node in incoming_nodes:
                if node.x < center_node.x:
                    if min_distance == -1 or abs(center_node.y - node.y) < min_distance:
                        min_distance = abs(center_node.y - node.y)
                        selected_node = node
            incoming_nodes.remove(selected_node)
            sorted_nodes.append(selected_node)

            # find North node
            min_distance, selected_node = -1, None
            for node in incoming_nodes:
                if node.y > center_node.y:
                    if min_distance == -1 or abs(center_node.x - node.x) < min_distance:
                        min_distance = abs(center_node.x - node.x)
                        selected_node = node
            incoming_nodes.remove(selected_node)
            sorted_nodes.append(selected_node)

            #find East node
            min_distance, selected_node = -1, None
            for node in incoming_nodes:
                if node.x > center_node.x:
                    if min_distance == -1 or abs(center_node.y - node.y) < min_distance:
                        min_distance = abs(center_node.y - node.y)
                        selected_node = node
            incoming_nodes.remove(selected_node)
            sorted_nodes.append(selected_node)

            # find South node
            min_distance, selected_node = -1, None
            for node in incoming_nodes:
                if node.y < center_node.y:
                    if min_distance == -1 or abs(center_node.x - node.x) < min_distance:
                        min_distance = abs(center_node.x - node.x)
                        selected_node = node
            incoming_nodes.remove(selected_node)
            sorted_nodes.append(selected_node)

            if None in sorted_nodes:
                print("<<<Check road structure of intersection %s >>>" % self.id)
                sys.exit(0)

        if len(incoming_nodes) == 3:            
            # check W, E or N, S
            # first check location:
            flag_W, flag_E = False, False
            flag_N, flag_S = False, False

            for node in incoming_nodes:
                if node.x < center_node.x:
                    flag_W = True
                else:
                    flag_E = True

                if node.y > center_node.y:
                    flag_N = True
                else:
                    flag_S = True
            if not (flag_W and flag_E and flag_N and flag_S):
                if flag_W:
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.x < center_node.x:
                            if min_distance == -1 or abs(center_node.y - node.y) < min_distance:
                                min_distance = abs(center_node.y - node.y)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)
                else:
                    sorted_nodes.append(None)

                if flag_N:
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.y > center_node.y:
                            if min_distance == -1 or abs(center_node.x - node.x) < min_distance:
                                min_distance = abs(center_node.x - node.x)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)
                else:
                    sorted_nodes.append(None)

                if flag_E:
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.x > center_node.x:
                            if min_distance == -1 or abs(center_node.y - node.y) < min_distance:
                                min_distance = abs(center_node.y - node.y)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)
                else:
                    sorted_nodes.append(None)

                if flag_S:
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.y < center_node.y:
                            if min_distance == -1 or abs(center_node.x - node.x) < min_distance:
                                min_distance = abs(center_node.x - node.x)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)
                else:
                    sorted_nodes.append(None)

            else:
                distance_to_W_E = []
                distance_to_N_S = []
                for node in incoming_nodes:
                    distance_to_W_E.append(abs(node.y-center_node.y))
                    distance_to_N_S.append(abs(node.x-center_node.x))
                distance_to_W_E = sorted(distance_to_W_E)
                distance_to_N_S = sorted(distance_to_N_S)
                if distance_to_W_E < distance_to_N_S:
                    # find West node
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.x < center_node.x:
                            if min_distance == -1 or abs(center_node.y - node.y) < min_distance:
                                min_distance = abs(center_node.y - node.y)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)

                    #find East node
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.x > center_node.x:
                            if min_distance == -1 or abs(center_node.y - node.y) < min_distance:
                                min_distance = abs(center_node.y - node.y)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)

                    if incoming_nodes[0].y > center_node.y:
                        # the last node is North
                        sorted_nodes.insert(1, incoming_nodes[0])
                        sorted_nodes.append(None)
                    else:
                        # the last node is South
                        sorted_nodes.insert(1, None)
                        sorted_nodes.append(incoming_nodes[0])

                else:
                    # find North node
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.y > center_node.y:
                            if min_distance == -1 or abs(center_node.x - node.x) < min_distance:
                                min_distance = abs(center_node.x - node.x)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)

                    # find South node
                    min_distance, selected_node = -1, None
                    for node in incoming_nodes:
                        if node.y < center_node.y:
                            if min_distance == -1 or abs(center_node.x - node.x) < min_distance:
                                min_distance = abs(center_node.x - node.x)
                                selected_node = node
                    incoming_nodes.remove(selected_node)
                    sorted_nodes.append(selected_node)

                    if node.x < center_node.x:
                        # the last node is West
                        sorted_nodes.insert(0, incoming_nodes[0])
                        sorted_nodes.insert(2, None)
                    else:
                        # the last node is East
                        sorted_nodes.insert(0, None)
                        sorted_nodes.insert(2, incoming_nodes[0])

        def getLanesArray(node, center_node):
            in_lanes_info, out_lanes_info = None, None
            if GloVars.config['simulator'] == 'SUMO':
                net = sumolib.net.readNet('./traffic-sumo/%s' % GloVars.config['net'])
                node_ = net.getNode(node.node_id)
                center_node_ = net.getNode(center_node.node_id)
                edges = node_.getOutgoing() 
                edges += node_.getIncoming()
                for edge in edges:
                    if edge.getToNode() == center_node_:
                        in_lanes_info = [{'id': lane.getID(), 'length': lane.getLength(), 'max_allowed_speed': lane.getSpeed()} for lane in edge.getLanes()]
                    if edge.getFromNode() == center_node_:
                        out_lanes_info = [{'id': lane.getID(), 'length': lane.getLength(), 'max_allowed_speed': lane.getSpeed()} for lane in edge.getLanes()]
            elif GloVars.config['simulator'] == 'CityFlow':
                config_CF = json.load(open(GloVars.config['config_file'], 'r'))
                road_structure = json.load(open(config_CF['dir'] + config_CF['roadnetFile'], 'r'))
                for road in road_structure['roads']:
                    if road['startIntersection'] == node.node_id and road['endIntersection'] == center_node.node_id:
                        distance = math.sqrt((road['points'][0]['x'] - road['points'][1]['x'])**2 + (road['points'][0]['y'] - road['points'][1]['y'])**2)
                        in_lanes_info = []
                        for idx, lane in enumerate(road['lanes']):
                            in_lanes_info.append({
                                'id': "%s_%d" % (road['id'], idx),
                                'length': distance,
                                'max_allowed_speed': float(lane['maxSpeed'])
                            })
                    if road['startIntersection'] == center_node.node_id and road['endIntersection'] == node.node_id:
                        distance = math.sqrt((road['points'][0]['x'] - road['points'][1]['x'])**2 + (road['points'][0]['y'] - road['points'][1]['y'])**2)
                        out_lanes_info = []
                        for idx, lane in enumerate(road['lanes']):
                            out_lanes_info.append({
                                'id': "%s_%d" % (road['id'], idx),
                                'length': distance,
                                'max_allowed_speed': lane['maxSpeed']
                            })
            return in_lanes_info, out_lanes_info

        road_structure = {}
        if sorted_nodes[0] != None:
            road_structure['west_road_in'], road_structure['west_road_out'] =  getLanesArray(sorted_nodes[0], center_node)
        if sorted_nodes[1] != None:
            road_structure['north_road_in'], road_structure['north_road_out'] =  getLanesArray(sorted_nodes[1], center_node)
        if sorted_nodes[2] != None:
            road_structure['east_road_in'], road_structure['east_road_out'] =  getLanesArray(sorted_nodes[2], center_node)
        if sorted_nodes[3] != None:
            road_structure['south_road_in'], road_structure['south_road_out'] =  getLanesArray(sorted_nodes[3], center_node)

        return road_structure

    def logStep(self, episode):
        def getQueueLength(self):
            queue_length = [];
            for lane in self.lanes_id:
                q = 0
                vehs = traci.lane.getLastStepVehicleIDs(lane)
                for veh in vehs:
                    if traci.vehicle.getSpeed(veh) < 5:
                        q += traci.vehicle.getLength(veh)
                queue_length.append(q)
            return queue_length

        log_ = {
            'ep': episode,
            'step': GloVars.step,
            'id': self.id,
            'CO2_emission': [traci.lane.getCO2Emission(lane) for lane in self.lanes_id],
            'CO_emission': [traci.lane.getCOEmission(lane) for lane in self.lanes_id],
            'fuel_consumption': [traci.lane.getFuelConsumption(lane) for lane in self.lanes_id],
            'num_halting_vehs': [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes_id],
            'speed': [traci.lane.getLastStepMeanSpeed(lane) for lane in self.lanes_id],
            'occupancy': [traci.lane.getLastStepOccupancy(lane) for lane in self.lanes_id],
            'num_vehs': [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes_id],
            'waiting_time': [traci.lane.getWaitingTime(lane) for lane in self.lanes_id],
            'queue_length': self.getQueueLength()
        }
        df = pd.DataFrame([log_])
        
        if not os.path.exists(GloVars.config['log_folder']):
            os.makedirs(GloVars.config['log_folder'])
        log_folder = '%s/log_per_intersection.csv' % GloVars.config['log_folder']
        if not os.path.isfile(log_folder):
            df.to_csv(log_folder, header='column_names', index=False)
        else: # else it exists so append without writing the header
            df.to_csv(log_folder, mode='a', header=False, index=False)
   
    def getState(self):
        """
            return the current state of the intersection:
            state = {
                'road_structure': {
                    'west_road_in': [{'id': 'W_lane_0', 'length': 100, 'width': 4, 'max_allowed_speed': 50}],
                    ...
                },
                'vehicles':[
                    {
                        'id': 'veh_0',
                        'lane': 'W_lane_0',
                        'distance_from_lane_start': 20,
                        'speed': 10,
                        'length': 10,
                        'waiting_time': 0.6,
                    },
                ]
                'current_phase_index': 0
            }
        """
        vehs = []
        if GloVars.config['simulator'] == "SUMO":
            for lane in self.lanes_id:
                for veh in traci.lane.getLastStepVehicleIDs(lane):
                    vehs.append({
                        'id': veh,
                        'lane': lane,
                        'distance_from_lane_start': traci.vehicle.getLanePosition(veh),
                        'speed': traci.vehicle.getSpeed(veh),
                        'max_speed': traci.vehicle.getMaxSpeed(veh),
                        'length': traci.vehicle.getLength(veh),
                        'waiting_time': traci.vehicle.getWaitingTime(veh)
                    })
            current_phase_index = traci.trafficlight.getPhase(self.id)
        elif GloVars.config['simulator'] == 'CityFlow':
            vehs_id = GloVars.eng.get_vehicles()
            for lane_id in self.lanes_id:
                for veh_id in vehs_id:
                    veh = GloVars.eng.get_vehicle_info(veh_id)
                    if veh['drivable'] == lane_id:
                        vehs.append({
                            'id': veh_id,
                            'lane': lane_id,
                            'distance_from_lane_start': float(veh['distance']),
                            'speed': float(veh['speed']),
                            'length': 5, #TODO CityFlow has not supported,
                            'max_speed': 11.11, #TODO CityFlow has not supported,
                            'waiting_time': 0 #TODO CityFlow has not supported,
                        })
            current_phase_index = self.current_phase_index

        return {
            'road_structure': self.road_structure,
            'phase_description': self.phase_description,
            'vehicles': vehs,
            'current_phase_index': current_phase_index
        }

    def processControlStack(self, control_stack):
        if len(control_stack) <= 0:
            return
        for action in control_stack:
            if action['type'] == ActionType.CHANGE_TO_NEXT_PHASE:
                self.control_actions.extend([{'type': ActionType.YELLOW_PHASE, 'length': self.yellow_duration, 'executed': False},
                                            {'type': ActionType.CHANGE_TO_NEXT_PHASE, 'length': action['length'], 'executed': False}])
            elif action['type'] == ActionType.CHANGE_TO_PHASE:
                self.control_actions.extend([{'type': ActionType.YELLOW_PHASE, 'length': self.yellow_duration, 'executed': False},
                                            {'type': ActionType.CHANGE_TO_PHASE, 'phase_index': action['phase_index'], 'length': action['length'], 'executed': False}])
            elif action['type'] == ActionType.KEEP_PHASE:
                self.control_actions.extend([{'type': ActionType.KEEP_PHASE, 'length': action['length'], 'executed': False}])
            else:
                print("<<< error in processControlStack >>>")
                sys.exit()

    def changeToNextPhase(self):
        """
            Call this function to change the traffic light to the next phase
        """
        if GloVars.config['simulator'] == 'SUMO':
            current_phase_ = traci.trafficlight.getPhase(self.id)
            if current_phase_ >= self.number_of_phases - 1:
                traci.trafficlight.setPhase(self.id, 0)
                self.current_phase_index = 0
            else:
                traci.trafficlight.setPhase(self.id, current_phase_ + 1)
                self.current_phase_index += 1
            traci.trafficlight.setPhaseDuration(self.id, MAX_INT)
        elif GloVars.config['simulator'] == 'CityFlow':
            current_phase_ = self.current_phase_index
            if current_phase_ >= self.number_of_phases - 1:
                GloVars.eng.set_tl_phase(self.id, 0)
                self.current_phase_index = 0
            else:
                GloVars.eng.set_tl_phase(self.id, current_phase_ + 1)
                self.current_phase_index += 1

    def changeToPhase(self, phase_idx):
        if GloVars.config['simulator'] == 'SUMO':
            traci.trafficlight.setPhase(self.id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.id, MAX_INT)
        elif GloVars.config['simulator'] == 'CityFlow':
            GloVars.eng.set_tl_phase(self.id, phase_idx)
            self.current_phase_index += 1

    def doAction(self):
        """
            This function is to execute the control_actions
            if not executed => executed
            -- length
            if length = 0 => pop
        """
        while len(self.control_actions) > 0:
            if self.control_actions[0]['executed'] is False:
                self.control_actions[0]['executed'] = True
                if self.control_actions[0]['type'] == ActionType.YELLOW_PHASE or self.control_actions[0]['type'] == ActionType.CHANGE_TO_NEXT_PHASE:
                    self.changeToNextPhase()
                elif self.control_actions[0]['type'] == ActionType.CHANGE_TO_PHASE:
                    self.changeToPhase(self.control_actions[0]['phase_index'])
            if self.control_actions[0]['length'] > 1:
                self.control_actions[0]['length'] -= 1
                break
            elif self.control_actions[0]['length'] == 1:
                self.control_actions[0]['length'] -= 1
                self.control_actions.pop(0)
                break
            elif self.control_actions[0]['length'] == 0:
                self.control_actions.pop(0)
                continue
            else:
                print("error in process control stack")
                sys.exit()

    def saveModel(self, ep=0):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.controller.model.save("%s/%s-%d.h5" % (self.folder, self.id, ep))
    
    def loadModel(self):
        try:
            #TODO: take care with 49
            self.controller.model.load_weights("%s/%s-%d.h5" % (self.folder, self.id, 49))
        except:
            print("No model to load %s/%s-%d.h5" % (self.folder, self.id, 49))

    def update(self, is_train=False, pretrain=False):
        """
            Call this function each time step
        """
        # TODO - new update function to suit for a stack of actions and more phases of a cycle
        # ....
        # every changing action has a yellow phase following
        # control_actions = [{'type': '1', 'length': 3, executed: True/False}, {'type': '1', 'length': 5, executed: True/False}]
        #                    => change -> keep this phase (yellow phase) for 3 seconds        => change -> keep this phase (green/red) for 5 seconds

        #   if len = 0 => no action in queue:
        #       get action by state
        #       process action & add into the control_actions
        #   if len >= 0 => have action in queue:
        #       if not yet execute => execute by +1 phase index
        #       -1 time length
        #           if time length == 0:
        #               that means finish the cycle => delete in queue

        #   if len = 0 => no action in queue:
        #       get action by state
        #       if action = True:
        #           change & add to queue including yellow and next cycle
        #       else:
        #           add to queue next cycle
        #   if len >= 0 => have action in queue:
        #       -1 time length
        #           if time length == 0:
        #               that means finish the cycle => delete in queue
        #                   check if has just deleted yellow phase (still have action in action_queue) => change phase
        control_stack = []
        if len(self.control_actions) <= 0:
            cur_state = self.getState()
            if is_train:
                if pretrain or (random.uniform(0, 1) <= GloVars.EXPLORE_PROBABILITY):
                    action, control_stack = self.controller.randomAction(cur_state)
                else:
                    action, control_stack = self.controller.makeAction(cur_state)

                # log last_state, last_action, reward, cur_state
                if (self.last_processed_state is not None) and (self.last_action is not None):
                    # compute reward
                    reward = self.controller.computeReward(cur_state, self.historical_data)
                    self.historical_data = self.controller.logHistoricalData(cur_state, control_stack[0]['type'])
                    self.controller.exp_memory.add([self.last_processed_state, self.last_action, reward, self.controller.processState(cur_state)])
                    # plot reward
                    if not pretrain:
                        with self.writer.as_default():
                            tf.summary.scalar('reward', reward, step=GloVars.step)

                self.last_processed_state, self.last_action = self.controller.processState(cur_state), action
            else:
                action, control_stack = self.controller.makeAction(cur_state)

        self.processControlStack(control_stack)
        self.doAction()

    def replay(self):
        self.controller.replay()