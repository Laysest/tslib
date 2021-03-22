from RLAgent import RLAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

ACTION_SPACE = 2
STATE_SPACE = 2

class SimpleRL(RLAgent):
    def build_model(self):
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