import numpy as np
from lib.data_parsing import DataParser
from lib.nn_model import NNModel

if __name__ == '__main__':
    split_data = True
    data_parser = DataParser('btcusd.csv')
    trainX, trainY = data_parser.generate_training_sets_auto(overlap_split=split_data,
                                                            split_months=6,
                                                            past_displacement=14, 
                                                            future_displacement=1)    
    model = NNModel()
    model.add_training_data(trainX=trainX, trainY=trainY, scaler = data_parser.scaler, is_split=split_data)
    model.addLayers(number=2, units=1024, activation='tanh', dropout=0.35)
    model.compile()

    model.train(epochs=2000, is_split=split_data)
    
    past_data = np.asarray(trainX[len(trainX)-1]).astype(np.float32)
    print(model.predict(past_data=past_data, future_displacement=7))


if __name__ == 'e':
    # Define autoencoder model
    model = Sequential()
    model.add(LSTM(32, activation='tanh', input_shape=(trainX[0].shape[1], trainX[0].shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16, activation='tanh', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY[0].shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Fit model
    for i in range(len(trainX)-1):
        model.fit(trainX[i], trainY[i], epochs=1000, batch_size=16, validation_split=0.1, verbose=1)
    # Plot
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.show()

    n_predict = 7
    forecast = model.predict(trainX[len(trainX)-2][-n_predict:])
    forecast_copies = np.repeat(forecast, trainX[len(trainX)-2][0].shape[1], axis=-1)
    scaler = data_parser.scaler
    y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
    print(y_pred_future)
    #print(trainX[-n_predict:])

