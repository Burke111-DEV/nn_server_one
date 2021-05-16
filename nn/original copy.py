if __name__ == '__man__':
    df = pd.read_csv('btcusd.csv') # 

    cols = list(df)[1:5] # Names of variables used
    df_for_training = df[cols].astype(float)

    # Scales the data.
    # LSTM uses sigmoid and tanh, which are sensitive to magnitude.
    # Normalising the data by scaling down high magnitudes improves performance.
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)

    # Training data
    # trainX = training set
    # trainY = prediction
    trainX, trainY = [], []

    # Distance in any direction. Past is length of trainX, future is length of trainY.
    n_future = 1
    n_past = 14

    # Generate training data
    for i in range(n_past, len(df_for_training_scaled) - n_future+1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training_scaled.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
    trainX, trainY = np.array(trainX), np.array(trainY)

    # Define autoencoder model
    model = Sequential()
    model.add(LSTM(256, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='tanh', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Fit model
    history = model.fit(trainX, trainY, epochs=20, batch_size=16, validation_split=0.1, verbose=1)
    # Plot
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.show()

    n_predict = 7
    forecast = model.predict(trainX[-n_predict:])
    forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
    print(y_pred_future)
    #print(trainX[-n_predict:])

