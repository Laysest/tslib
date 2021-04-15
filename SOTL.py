from controller import Controller, ActionType
from glo_vars import GloVars

traci = GloVars.traci

MIN_GREEN_VEHICLE = 20
MAX_RED_VEHICLE = 30

class SOTL(Controller):
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
            num_veh_ordered.append(traci.lane.getLastStepVehicleNumber(lane))
        
        return current_logic, num_veh_ordered


    def makeAction(self, state_):
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

    def actionType(self):
        return ActionType.CHANGING_KEEPING