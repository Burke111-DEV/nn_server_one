import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import os.path

class NNModel():
    def __init__(self, load_model_name=-1):
        if load_model_name == -1:
            self.model = Sequential()
        else:
            self.model = load_model("models/"+load_model_name)

    def add_training_data(self, trainX, trainY, scaler, is_split=False):
        self.trainX = trainX
        self.trainY = trainY
        if is_split:
            self.shapeX = (trainX[0].shape[0], trainX[0][0].shape[0], trainX[0][0].shape[1])
            self.shapeY = (trainY[0].shape[0], trainY[0][0].shape[0])
        else:
            self.shapeX = trainX.shape
            self.shapeY = trainY.shape
        self.scaler = scaler

    def addLayer(self, units=32, activation='tanh', return_sequence=True, dropout=0.2):
        # print("_________________________________________________________________")
        # print((self.shapeX[1], self.shapeX[2]))
        # print("_________________________________________________________________")

        self.model.add(  LSTM(units,
                        activation=activation, 
                        input_shape=(self.shapeX[1], self.shapeX[2]), 
                        return_sequences=return_sequence))
        self.model.add(Dropout(dropout))

    def addLayers(self, number, units=32, activation='tanh', dropout=0.2):
        for i in range(number-1):
            self.addLayer(units=units, activation=activation, return_sequence=True, dropout=dropout)
        self.addLayer(units=int(units/2), activation=activation, return_sequence=False)
        self.model.add(Dense(self.shapeY[1]))
    
    def compile(self):
        print("_________________________________________________________________")
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()
    
    def train(self, epochs=1, batch_size=16, validation_split=0.1, verbose=1, is_split=False):
        if is_split:
            # for h in range(len(self.trainX)-1):
            #     for i in range(len(self.trainX[h])-1):
            #         self.model.fit( np.asarray(self.trainX[h][i]).astype('float32'), 
            #                         np.asarray(self.trainY[h][i]).astype('float32'), 
            #                         epochs=epochs, 
            #                         batch_size=batch_size, 
            #                         validation_split=validation_split, 
            #                         verbose=verbose)

            for i in range(len(self.trainX)-1):
                self.model.fit( np.asarray(self.trainX[i]).astype('float32'), 
                                np.asarray(self.trainY[i]).astype('float32'), 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                validation_split=validation_split, 
                                verbose=verbose)
        else:
            self.model.fit( self.trainX, 
                            self.trainY, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_split=validation_split, 
                            verbose=verbose)

    def predict(self, past_data, future_displacement=1):
        forecast = self.model.predict(past_data[-future_displacement:])
        fc_cpy = np.repeat(forecast, self.shapeX[2], axis=-1)
        return self.scaler.inverse_transform(fc_cpy)[:,0]

    def save(self, name):
        self.model.save("models/"+name)
