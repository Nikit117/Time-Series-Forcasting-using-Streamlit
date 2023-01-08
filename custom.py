import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
st.header("Time Series Data Forcasting using CNN-LSTM, N-BEATS and Ensemble Model")
st.write("Made by: Nikit Kashyap")
number = st.number_input('Enter the index of the date/time column', min_value=0, max_value=100, value=0, step=1)
df = st.file_uploader("upload file", type={"csv", "txt"})
if df is not None:
    df = pd.read_csv(df, parse_dates=True, index_col=[number])
    st.write(df)
    option2 = st.selectbox(
    'select the column to be predicted',
    df.keys())
    st.write('You selected:', option2)
    prices = pd.DataFrame(df[option2]).rename(columns={option2: "Price"})
    st.write(prices.head())
    st.write("The shape of the dataset is: ", prices.shape)
    st.write("The number of missing values in the dataset is: ", prices.isnull().sum())
    windows = st.number_input('Enter the number of days to be used for prediction', min_value=1, max_value=100, value=1, step=1)
    horizon = 1
    for i in range(windows): # Shift values for each step in WINDOW_SIZE
      prices[f"Price+{i+1}"] = prices["Price"].shift(periods=i+1)
    X = prices.dropna().drop("Price", axis=1).astype(np.float32) 
    y = prices.dropna()["Price"].astype(np.float32)
    st.write(X.head(10))
    agree = st.checkbox('Do you want to see the data in the form of a graph?')
    if agree:
        st.line_chart(y)

    val = st.slider('Percentage Input for Traning/Testing',min_value=0, max_value=100)
    val = val/100
    split_size = int(len(X) * val)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    st.write(val*100,"%"," of the dataset is used for training and ",100-val*100," is used for testing")

# 1. Turn train and test arrays into tensor Datasets
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

# 2. Combine features & labels
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

# 3. Batch and prefetch for optimal performance
    BATCH_SIZE  = st.slider('Enter BATCH_SIZE', min_value=1, max_value=1000)
    if BATCH_SIZE is None:
        BATCH_SIZE = 128
# taken from Appendix D in N-BEATS paper
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def create_model_checkpoint(model_name, save_path="model_experiments"):
      return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), # create filepath to save model
                                                verbose=0, # only output a limited amount of text
                                                save_best_only=True) # save only the best model to file
    
    def evaluate_preds(y_true, y_pred):
      # Make sure float32 (for metric calculations)
      y_true = tf.cast(y_true, dtype=tf.float32)
      y_pred = tf.cast(y_pred, dtype=tf.float32)
    
      # Calculate various metrics
      mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
      mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
      rmse = tf.sqrt(mse)
      mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    
      # Account for different sized metrics (for longer horizons, reduce to single number)
      if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
    
      return {"mae": mae.numpy(),
              "mse": mse.numpy(),
              "rmse": rmse.numpy(),
              "mape": mape.numpy(),
              }

    def make_preds(model, input_data):
      forecast = model.predict(input_data)
      return tf.squeeze(forecast)
    
    Into_the_future = st.number_input('Enter the number of days to be predicted', min_value=1, max_value=100, value=1, step=1, key='Into_the_future')
    
    def make_future_forecast(values, model, into_future, window_size=windows) -> list:
    
      # 2. Make an empty list for future forecasts/prepare data to forecast on
        future_forecast = []
        last_window = values[-windows:] # only want preds from the last window (this will get updated)
    
      # 3. Make INTO_FUTURE number of predictions, altering the data which gets predicted on each time 
        for _ in range(into_future):
        
        # Predict on last window then append it again, again, again (model starts to make forecasts on its own forecasts)
            future_pred = model.predict(tf.expand_dims(last_window, axis=0))
            print(f"Predicting on: \n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")
        
        # Append predictions to future_forecast
            future_forecast.append(tf.squeeze(future_pred).numpy())
        # print(future_forecast)
    
        # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
            last_window = np.append(last_window, future_pred)[-windows:]
      
        return future_forecast
    
    models = st.multiselect('Select the models to be used', ['N-BEATS', 'CNN-LSTM', 'Ensemble'])
    if models is not None:
        st.write('You selected:', models)
    if 'CNN-LSTM' in models:
        st.write('CNN-LSTM')
        Epochs1  = st.slider('Enter EPOCHS', min_value=1, max_value=1000, value=0, step=1, key='Epochs1')
        from tensorflow.keras.layers import Conv1D, Conv2D, Input, Dense, Flatten, Dropout, LSTM, Reshape, Concatenate, TimeDistributed, Bidirectional,Lambda,GlobalMaxPooling1D
        from tensorflow.keras.models import Model
        model_1 = tf.keras.Sequential([
              tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=1)),
              tf.keras.layers.LSTM(200, activation="relu"),
              tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=1)),
               tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, activation="relu")),
              tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=1)),
              tf.keras.layers.Conv1D(filters=128, kernel_size=5,strides=1, padding="causal",
                          activation="relu"),
              tf.keras.layers.GlobalMaxPooling1D(),
              tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=1)),
              tf.keras.layers.Conv1D(filters=128, kernel_size=5,strides=1, padding="causal",
                          activation="relu"),
              tf.keras.layers.GlobalMaxPooling1D(),
              tf.keras.layers.Dense(128,activation="relu"),
              tf.keras.layers.Dense(64,activation="relu"),
              tf.keras.layers.Dense(1)
           ],name="model_1")
        model_1.compile(loss="mae",optimizer="Adam",metrics=["mae"])
        model_1.fit(train_dataset, epochs=Epochs1, verbose = 1, batch_size = 128,validation_data=test_dataset,callbacks=[create_model_checkpoint(model_name=model_1.name)])
        model_1 = tf.keras.models.load_model("model_experiments/model_1")
        cnn_rnn_preds = make_preds(model_1, test_dataset)
        # st.write(cnn_rnn_preds)
        st.write("CNN-LSTM Model Performance:")
        st.write(evaluate_preds(y_true=tf.squeeze(y_test), y_pred=cnn_rnn_preds))
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(len(y_test)), y_test, 'b', label="actual")
        plt.plot(np.arange(len(y_test)), cnn_rnn_preds, 'r', label="forecast")
        plt.title("CNN-RNN vs Test data", fontsize=24)
        plt.xlabel("Time")
        plt.ylabel("predicted")
        plt.legend()
        st.pyplot(plt)
        future_forecast = make_future_forecast(values = y, model = model_1, into_future = Into_the_future, window_size = windows)
        st.write("Forecasted values for next ",Into_the_future," days are: ")
        st.write(future_forecast)
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(len(y)), y, 'b', label="actual")
        plt.plot(np.arange(len(y), len(y) + len(future_forecast)), future_forecast, 'r', label="forecast")
        plt.title("CNN-RNN Forecasting for next days", fontsize=20)
        plt.xlabel("Time")
        plt.ylabel("Forecasted")
        plt.legend()
        st.pyplot(plt)
    
    if 'N-BEATS' in models:
        st.write('N-BEATS')
        Epochs2  = st.slider('Enter EPOCHS', min_value=1, max_value=1000,key='Epochs2', value=0, step=1)
        # Create NBeatsBlock custom layer 
        class NBeatsBlock(tf.keras.layers.Layer):
            def __init__(self, # the constructor takes all the hyperparameters for the layer
                   input_size: int,
                   theta_size: int,
                   horizon: int,
                   n_neurons: int,
                   n_layers: int,
                   **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
                super().__init__(**kwargs)
                self.input_size = input_size
                self.theta_size = theta_size
                self.horizon = horizon
                self.n_neurons = n_neurons
                self.n_layers = n_layers
    
                # Block contains stack of 4 fully connected layers each has ReLU activation
                self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
                # Output of block is a theta layer with linear activation
                self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")
    
            def call(self, inputs): # the call method is what runs when the layer is called 
                x = inputs 
                for layer in self.hidden: # pass inputs through each hidden layer 
                    x = layer(x)
                theta = self.theta_layer(x) 
                # Output the backcast and forecast from theta
                backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
                return backcast, forecast
        
        dummy_nbeats_block_layer = NBeatsBlock(input_size=windows, 
                                           theta_size=windows+horizon, # backcast + forecast 
                                           horizon=horizon,
                                           n_neurons=128,
                                           n_layers=4)
                            
        N_EPOCHS = Epochs2 # called "Iterations" in Table 18
        N_NEURONS = 512 # called "Width" in Table 18
        N_LAYERS = 4
        N_STACKS = 30
    
        INPUT_SIZE = windows * horizon # called "Lookback" in Table 18
        THETA_SIZE = INPUT_SIZE + horizon # backcast + forecast
    
        # Create NBeatsBlock custom layer
        tf.random.set_seed(42)
        import tensorflow as tf
        from tensorflow.keras import layers
    # 1. Setup N-BEATS Block layer
        nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                     theta_size=THETA_SIZE,
                                     horizon=horizon,
                                     n_neurons=N_NEURONS,
                                     n_layers=N_LAYERS,
                                     name="InitialBlock")
    
    # 2. Create input to stacks
        stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")
    
    # 3. Create initial backcast and forecast input (backwards predictions are referred to as residuals in the paper)
        backcast, forecast = nbeats_block_layer(stack_input)
    # Add in subtraction residual link 
        residuals = layers.subtract([stack_input, backcast], name=f"subtract_00") 
    
    # 4. Create stacks of blocks
        for i, _ in enumerate(range(N_STACKS-1)): # first stack is already creted in (3)
    
      # 5. Use the NBeatsBlock to calculate the backcast as well as block forecast
          backcast, block_forecast = NBeatsBlock(
          input_size=INPUT_SIZE,
          theta_size=THETA_SIZE,
          horizon=horizon,
          n_neurons=N_NEURONS,
          n_layers=N_LAYERS,
          name=f"NBeatsBlock_{i}"
          )(residuals) # pass it in residuals (the backcast)
    
      # 6. Create the double residual stacking
          residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}") 
          forecast = layers.add([forecast, block_forecast], name=f"add_{i}")
    
    # 7. Put the stack model together
        model_2 = tf.keras.Model(inputs=stack_input, 
                             outputs=forecast, 
                             name="model_2_N-BEATS")
    
    # 8. Compile with MAE loss and Adam optimizer
        model_2.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=["mae", "mse"])
    
    # 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
        model_2.fit(train_dataset,
                epochs=N_EPOCHS,
                validation_data=test_dataset,
                verbose=2, # prevent large amounts of training outputs
                # callbacks=[create_model_checkpoint(model_name=stack_model.name)] # saving model every epoch consumes far too much time
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])
    
        model_2_preds = make_preds(model_2, test_dataset)
        st.write("N-BEATS Model Performance:")
        st.write(evaluate_preds(y_true=y_test, y_pred=model_2_preds))
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(len(y_test)), y_test, 'b', label="actual")
        plt.plot(np.arange(len(y_test)), model_2_preds, 'r', label="forecast")
        plt.title("N-BEATS", fontsize=24)
        plt.xlabel("Time")
        plt.ylabel("predicted")
        plt.legend()
        st.pyplot(plt)
        future_forecast2 = make_future_forecast(values = y, model = model_2, into_future = Into_the_future, window_size = windows)
        st.write("Forecasted values for next ",Into_the_future," days are: ")
        st.write(future_forecast2)
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(len(y)), y, 'b', label="actual")
        plt.plot(np.arange(len(y), len(y)+Into_the_future), future_forecast2, 'r', label="forecast")
        plt.title("N-BEATS Forecasting for next days", fontsize=24)
        plt.xlabel("Time")
        plt.ylabel("Forecasted")
        plt.legend()
        st.pyplot(plt)
    
    if 'Ensemble' in models:
        st.write('Ensemble')
        Epochs3  = st.slider('Enter EPOCHS', min_value=1, max_value=1000, value=0, step=1, key="Epochs3")
        import tensorflow as tf
        from tensorflow.keras import layers
        def get_ensemble_models(horizon=horizon, 
                            train_data=train_dataset,
                            test_data=test_dataset,
                            num_iter=10, 
                            num_epochs=100, 
                            loss_fns=["mae", "mse", "mape"]):
    
      # Make empty list for trained ensemble models
            ensemble_models = []
    
      # Create num_iter number of models per loss function
            for i in range(num_iter):
        # Build and fit a new model with a different loss function
              for loss_function in loss_fns:
                print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}")
    
          # Construct a simple model (similar to model_1)
                model = tf.keras.Sequential([
            # Initialize layers with normal (Gaussian) distribution so we can use the models for prediction
            # interval estimation later: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
                  layers.Dense(128, kernel_initializer="he_normal", activation="relu"), 
                  layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
                  layers.Dense(horizon)                                 
                ])
    
          # Compile simple model with current loss function
                model.compile(loss=loss_function,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae", "mse"])
          
          # Fit model
                model.fit(train_data,
                    epochs=num_epochs,
                    verbose=0,
                    validation_data=test_data,
                    # Add callbacks to prevent training from going/stalling for too long
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                patience=200,
                                                                restore_best_weights=True),
                               tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                    patience=100,
                                                                    verbose=1)])
          
          # Append fitted model to list of ensemble models
                ensemble_models.append(model)
    
            return ensemble_models # return list of trained models
        
        ensemble_models = get_ensemble_models(num_iter=5,
                                          num_epochs=Epochs3)
        def make_ensemble_preds(ensemble_models, data):
            ensemble_preds = []
            for model in ensemble_models:
                preds = model.predict(data) # make predictions with current ensemble model
                ensemble_preds.append(preds)
            return tf.constant(tf.squeeze(ensemble_preds))
        
        ensemble_preds = make_ensemble_preds(ensemble_models=ensemble_models,data=test_dataset)
        ensemble_results = evaluate_preds(y_true=y_test,
                                      y_pred=np.median(ensemble_preds, axis=0))
        st.write("Ensemble Model Performance:")
        st.write(ensemble_results)
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(len(y_test)), y_test, 'b', label="actual")
        plt.plot(np.arange(len(y_test)), np.median(ensemble_preds, axis=0), 'r', label="predicted")
        plt.title("Ensemble model", fontsize=24)
        plt.xlabel("Time")
        plt.ylabel("predicted")
        plt.legend()
        st.pyplot(plt)
        pred = []
        for models in range(len(ensemble_models)):
            pred.append(make_future_forecast(values = y, model = ensemble_models[models], into_future = Into_the_future, window_size = windows))
        pred = np.array(pred)
        pred = np.median(pred, axis=0)
        st.write("Forecasted values for next ",Into_the_future," days are: ")
        st.write(pred)
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(len(y)), y, 'b', label="actual")
        plt.plot(np.arange(len(y), len(y)+Into_the_future), pred, 'r', label="forecast")
        plt.title("Ensemble model Forecasting for next days", fontsize=24)
        plt.xlabel("Time")
        plt.ylabel("Forecasted")
        plt.legend()
        st.pyplot(plt)
    
    
