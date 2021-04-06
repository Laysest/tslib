from RLAgent import RLAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

ACTION_SPACE = 2
STATE_SPACE = 2

class SimplePhaseGate(RLAgent):
    def processState(self, state):
        """
            from general state returned from traffic light
            process to return (number_veh_on_green_lanes, number_veh_on_red_lanes)
        """

        num_veh_ordered = []
        for lane in state['lanes']:
            num_veh_ordered.append(state['traci'].lane.getLastStepVehicleNumber(lane))

        number_veh_on_green_lanes = 0
        number_veh_on_red_lanes = 0
        for i in range(len(num_veh_ordered)):
            if state['current_logic'][i] in ['r', 'R']:
                number_veh_on_red_lanes += num_veh_ordered[i]
            elif state['current_logic'][i] in ['g', 'G']:
                number_veh_on_green_lanes += num_veh_ordered[i]
            else:
                print("Error in getState in case of SimpleRL")

        return [number_veh_on_green_lanes, number_veh_on_red_lanes]
    
    def computeReward(self, state):
        reward = 0
        for lane in state['lanes']:
            reward -= state['traci'].lane.getLastStepHaltingNumber(lane)
        return reward

    def buildModel(self):
        """
            return the model in keras
        """
        model = Sequential()
        model.add(Dense(16, input_dim=STATE_SPACE))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(ACTION_SPACE))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model