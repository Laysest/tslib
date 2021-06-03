import sys
from controller import Controller, ActionType
from glo_vars import GloVars
import sumolib
traci = GloVars.traci

MIN_GREEN_VEHICLE = 10
MAX_RED_VEHICLE = 15

class MaxPressure(Controller):
    # pylint: disable=line-too-long invalid-name too-many-instance-attributes
    """
        The implementation of SOTL method
    """
    def __init__(self, config, tf_id):
        Controller.__init__(self)
        self.cycle_control = config['cycle_control']
        self.tf_id = tf_id
        self.lanes = traci.trafficlight.getControlledLanes(self.tf_id)
        self.lanes_unique = list(dict.fromkeys(self.lanes))

        # nodes, center = self.getNodesSortedByDirection()
        # self.outgoing_lanes = []
        # for edge in center.getOutgoing():
        #     for lane in edge.getLanes():
        #         self.outgoing_lanes.append(lane.getID())
        
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

    def processState(self, state):
        """
            from general state returned from traffic light
            process to return (current_logic, num_veh_ordered)
            current_logic: 'ggggrrrrgggg' shows status of traffic light
            num_veh_ordered: [1, 2, 1, 5, ...] shows number of vehicles on each lane by order
        """
        all_logic_ = traci.trafficlight.getAllProgramLogics(self.tf_id)[0]
        current_logic = all_logic_.getPhases()[all_logic_.currentPhaseIndex].state
        number_veh_on_green_lanes = 0
        number_veh_on_red_lanes = 0
        for lane in self.lanes_unique:
            idx = self.lanes.index(lane)
            if current_logic[idx] in ['r', 'R']:
                number_veh_on_red_lanes += traci.lane.getLastStepVehicleNumber(lane)
            elif current_logic[idx] in ['g', 'G']:
                number_veh_on_green_lanes += traci.lane.getLastStepVehicleNumber(lane)
            else:
                print("Error - do action during yellow phase")
                sys.exit()

        return number_veh_on_red_lanes, number_veh_on_green_lanes

    def makeAction(self, state):
        number_veh_on_red_lanes, number_veh_on_green_lanes = self.processState(state)
        if (number_veh_on_red_lanes > number_veh_on_green_lanes):
            return 1, [{'type': ActionType.CHANGE_PHASE, 'length': self.cycle_control, 'executed': False}]
        return 0, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]