import sys
import numpy as np
from controller import Controller, ActionType
from glo_vars import GloVars
traci = GloVars.traci

MIN_GREEN_VEHICLE = 10
MAX_RED_VEHICLE = 15

class MaxPressure(Controller):
    # pylint: disable=line-too-long invalid-name too-many-instance-attributes
    """
        The implementation of SOTL method
    """
    def __init__(self, config, road_structure, number_of_phases):
        Controller.__init__(self)
        self.cycle_control = config['cycle_control']
        self.incoming_lanes = [lane for k, road in road_structure.items() if 'in' in k for lane in road]
        self.number_of_phases = number_of_phases

    def processState(self, state):
        """
            from general state returned from traffic light
            process to return (current_logic, num_veh_ordered)
            current_logic: 'ggggrrrrgggg' shows status of traffic light
            num_veh_ordered: [1, 2, 1, 5, ...] shows number of vehicles on each lane by order
        """
        def get_number_vehicles_on_lane(vehs, lane_id):
            n = 0
            for veh in vehs:
                if veh['lane'] == lane_id:
                    n +=  1
            return n

        phase_pressure = []
        for idx, phase_des in enumerate(state['phase_description']):
            if idx % 2 != 0:
                continue
            pressure = 0
            approaching_lanes = []
            outgoing_lanes = []
            for item in phase_des:
                if item['from'] not in approaching_lanes:
                    pressure += get_number_vehicles_on_lane(state['vehicles'], item['from'])
                    approaching_lanes.append(item['from'])
                if item['to'] not in outgoing_lanes:
                    pressure -= get_number_vehicles_on_lane(state['vehicles'], item['to'])
                    outgoing_lanes.append(item['to'])
            phase_pressure.append(pressure)

        return phase_pressure

    def makeAction(self, state):
        phase_pressure = self.processState(state)
        action = np.argmax(phase_pressure)
        if 2*action == state['current_phase_index']:
            return action, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]
        return action, [{'type': ActionType.CHANGE_TO_PHASE, 'phase_index': action*2, 'length': self.cycle_control, 'executed': False}]
