import sys
from src.controller.controller import Controller, ActionType
from src.glo_vars import GloVars

traci = GloVars.traci

MIN_GREEN_VEHICLE = 10
MAX_RED_VEHICLE = 15

class SOTL(Controller):
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
        from src.environment.traffic_light import LightState
        current_phase_index = state['current_phase_index']
        next_phase_index = state['current_phase_index'] + 2 if state['current_phase_index'] + 2 < self.number_of_phases else 0 
        
        def get_light_states(phase_detail):
            light_states = {}
            for item in phase_detail:
                if item['from'] not in light_states:
                    light_states[item['from']] = 0
                if item['light_state'] == LightState.Green:
                    light_states[item['from']] += 1

            for key, val in light_states.items():
                if light_states[key] > 0:
                    light_states[key] = LightState.Green
                else:
                    light_states[key] = LightState.Red
            
            return light_states

        current_light_states = get_light_states(state['phase_description'][current_phase_index])
        next_light_states = get_light_states(state['phase_description'][next_phase_index])

        def get_number_vehicles_on_lane(vehs, lane):
            n = 0
            for veh in vehs:
                if veh['lane'] == lane['id']:
                    n +=  1
            return n

        number_veh_on_green_lanes = 0
        number_veh_on_red_lanes = 0
        for lane in self.incoming_lanes:
            if lane['id'] not in current_light_states.keys():
                continue
            if current_light_states[lane['id']] == LightState.Green:
                number_veh_on_green_lanes += get_number_vehicles_on_lane(state['vehicles'], lane)

            if (current_light_states[lane['id']] == LightState.Red) and (next_light_states[lane['id']] == LightState.Green):
                number_veh_on_red_lanes += get_number_vehicles_on_lane(state['vehicles'], lane)

        return number_veh_on_red_lanes, number_veh_on_green_lanes

    def makeAction(self, state):
        number_veh_on_red_lanes, number_veh_on_green_lanes = self.processState(state)
        if (number_veh_on_green_lanes < MIN_GREEN_VEHICLE and number_veh_on_red_lanes > MAX_RED_VEHICLE) \
                or (number_veh_on_green_lanes == 0 and number_veh_on_red_lanes > 0):
            return 1, [{'type': ActionType.CHANGE_TO_NEXT_PHASE, 'length': self.cycle_control, 'executed': False}]
        return 0, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': False}]
