import numpy as np
import time
from nn_lib.data_parsing import DataParser
from nn_lib.nn_model import NNModel

if __name__ == '__main__':
    split_data = True
    data_parser = DataParser('btcusd.csv')
    trainX, trainY = data_parser.generate_training_sets_auto(overlap_split=split_data,
                                                            split_months=6,
                                                            past_displacement=14, 
                                                            future_displacement=1)    
    model = NNModel()
    model.add_training_data(trainX=trainX, trainY=trainY, scaler = data_parser.scaler, is_split=split_data)
    model.addLayers(number=2, units=128, activation='tanh', dropout=0.35)
    model.compile()
    model.train(epochs=1200, is_split=split_data)
    
    model.save(str(np.round(time.time()*1000))+".h5")

    # past_data = np.asarray(trainX[len(trainX)-1]).astype(np.float32)
    # print(model.predict(past_data=past_data, future_displacement=7))