from controller import Controller

MIN_GREEN_VEHICLE = 20
MAX_RED_VEHICLE = 30

class SOTL(Controller):
    def make_action(self, state):
        print(state)
        # number_veh_lane_green = 0
        # number_veh_lane_red = 0

        # for lane in LANE_PERMISSION[str(current_phase)]['red']:
        #     number_veh_lane_red += obs["%s_total" % lane]
        # for lane in LANE_PERMISSION[str(current_phase)]['green']:
        #     number_veh_lane_green += obs["%s_total" % lane]
        # if (number_veh_lane_green < MIN_GREEN_VEHICLE and number_veh_lane_red > MAX_RED_VEHICLE) or (number_veh_lane_green == 0 and number_veh_lane_red > 0) :        
        #     return True
        return False