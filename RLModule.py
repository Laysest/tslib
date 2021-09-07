import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

class RLModule():
    def buildModel(type='DQN', input_shape=(64,), action_space=2):
        model = Sequential()

        # feature input
        if len(input_shape) == 1:
            model.add(Dense(16, input_dim=input_shape(0)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(action_space, activation='linear'))
        # image input
        elif len(input_shape) == 3:
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(action_space, activation='linear'))
        
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model