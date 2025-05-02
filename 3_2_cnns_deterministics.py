# -*- coding: utf-8 -*-
"""3.2 CNNs Deterministics.ipynb

# **CNN Deterministic**

Imports
"""

import tensorflow as tf
import tensorflow_probability as tfp
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Layer, MaxPooling1D, Dropout, Dense, GlobalAveragePooling1D, BatchNormalization, LSTM, LeakyReLU, ELU, Bidirectional, Multiply, Permute, Reshape, RepeatVector, Activation, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam

"""First, we will train a standard deterministic CNN model as a baseline before implementing Bayesian neural networks. In a deterministic neural network, there is only one weight, learned through backpropagation. In a neural network with weight uncertainty, each weight is represented by a probability distribution, and the parameters of this distribution are learned through backpropagation.

Loading Data
"""

df_cleaned = pd.read_csv("dataframe_final.csv", sep=",")

X_reduced = df_cleaned.drop(columns=['Tasa interanual del IPI', 'Valor'])
y = df_cleaned['Valor']  # Target variable

"""The results were more suitable using StandardScaler because the data distribution is Gaussian, while with MinMaxScaler the outliers were highly sensitive and were not represented correctly."""

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_reduced)

scaler_y = StandardScaler()
y = np.array(y)  # Convert to NumPy array
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))  # Reshaping required as y is a 1D vector

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Transform data for CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Add channel, (N samples, features) -> (N samples, features, dimension)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

def evaluate_model(y_test, y_pred, history=None):
    # Calculate basic metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    # Calculate range of real data
    Min = y_test.min()
    Max = y_test.max()
    Range = Max - Min

    if Range == 0:
        print("The range of the data is zero. Relative percentages cannot be calculated.")
    else:
        rmse_percentage = (rmse * 100) / Range
        print(f'RMSE in %: {rmse_percentage:.4f}%')

    # Calculate MAPE in %
    if np.any(y_test == 0):
        print("MAPE calculation cannot be performed because y_test contains zero values.")
    else:
        mape = (mae * 100) / Range
        print(f'MAPE in %: {mape:.4f}%')

    # Graphs (if there is training history)
    if history is not None:
        plt.figure(figsize=(8, 4))

        # Loss graph during training
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Loss', linestyle='-')
        plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
        plt.title('Loss Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # MAE graph during training
        if 'mae' in history.history and 'val_mae' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='MAE', linestyle='-')
            plt.plot(history.history['val_mae'], label='Val MAE', linestyle='--')
            plt.title('MAE Evolution')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = tf.keras.layers.ReLU()(x)  # Using ReLU activation
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# LSTM Layer for sequence modeling
x = LSTM(64, activation='tanh', return_sequences=True)(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = tf.keras.layers.ReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = tf.keras.layers.ReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.summary()

"""**CNN + LSTM**

Next, we will add LSTM (Long Short-Term Memory), a variant designed to capture long-term dependencies in sequences of data, thus solving the problem of not having access to the IPI in addition to improving the generalization of our model.

LSTMs work with gates that control the flow of information:
*   **Forget gate:** Decides which information from memory is kept or discarded.
*   **Input gate:** Decides which new information is added to memory.
*   **Output gate:** Determines which information is passed to the next layer or output.

These gates help maintain memory over long periods, allowing LSTMs to remember relevant information over long durations, whereas traditional RNNs cannot.

LSTMs are widely used for tasks like time series prediction, machine translation, sentiment analysis, and music generation due to their ability to handle sequential and temporally dependent data effectively.

In summary, LSTMs are very useful for working with long sequences.

**Traditional LSTM:** Processes the input sequence in one direction (typically left to right) to learn past dependencies.
"""

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = tf.keras.layers.ReLU()(x)  # Using ReLU activation
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# LSTM Layer for sequence modeling
x = LSTM(64, activation='tanh', return_sequences=True)(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = tf.keras.layers.ReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = tf.keras.layers.ReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

history_df = pd.DataFrame(history.history)

# Save the DataFrame as CSV
history_df.to_csv('modelLSTM_Relu_.csv', index=False)

# Save the complete model
model.save('model_LSTM_Relu.h5')

# Predictions on the test set
y_pred = model.predict(X_test_cnn)

# Evaluate the model
evaluate_model(y_test, y_pred, history=history)

"""**Bidirectional LSTM:**

The Bidirectional variant of a recurrent neural network, such as Bidirectional LSTM, is an extension that allows the model to process the data sequence in both directions: forward (from start to end) and backward (from last element to first). This provides the network with more context of the sequence as it considers both previous (like a standard LSTM) and future information from the sequence.
"""

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = tf.keras.layers.ReLU()(x)  # Using ReLU activation
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# LSTM Layer for sequence modeling
x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = tf.keras.layers.ReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = tf.keras.layers.ReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4, verbose=1)

history = model.fit(X_train_cnn, y_train, epochs=125, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

history_df = pd.DataFrame(history.history)

# Save the DataFrame as CSV
history_df.to_csv('modelLSTM_BI_Relu_.csv', index=False)

# Save the complete model
model.save('model_LSTM_BI_Relu.h5')

# Predictions on the test set
y_pred = model.predict(X_test_cnn)

# Evaluate the model
evaluate_model(y_test, y_pred, history=history)

"""**Leaky ReLU**

We will introduce the Leaky ReLU (Leaky Rectified Linear Unit) into our previously created model. This is a variant of the ReLU activation function applied in the previous models, but this variant introduces a small slope for negative inputs, preventing neurons from becoming inactive (a problem known as dead neurons). The function is defined as 𝑓(𝑥)=𝑥 for 𝑥>0, and 𝑓(𝑥)=𝛼𝑥 for 𝑥≤0, where 𝛼 is a small constant. We will try different values for 𝛼 to see the result. This allows negative inputs to contribute slightly to the gradient, improving training. The larger the value of 𝛼, the more it differs from conventional ReLU.

Modifying the value of LeakyReLU for a smaller slope makes it more similar to a conventional ReLU.

LSTM
"""

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # Using LeakyReLU
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# BiLSTM Layer for sequence modeling
x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4, verbose=1)

history = model.fit(X_train_cnn, y_train, epochs=125, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

history_df = pd.DataFrame(history.history)
history_df.to_csv('modelLSTM_LeakyReLU_.csv', index=False)
model.save('model_LSTM_LeakyReLU.h5')

# Prediccions sobre el conjunt de test
y_pred = model.predict(X_test_cnn)

# Evaluar el model
evaluate_model(y_test, y_pred, history=history)

"""Bidireccional LSTM"""

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# BiLSTM Layer for sequence modeling
x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4, verbose=1)

history = model.fit(X_train_cnn, y_train, epochs=125, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

history_df = pd.DataFrame(history.history)
history_df.to_csv('modelLSTM_LeakyReLU_.csv', index=False)
model.save('model_LSTM_LeakyReLU.h5')

# Predictions on the test set
y_pred = model.predict(X_test_cnn)

# Evaluate the model
evaluate_model(y_test, y_pred, history=history)

"""Bidirectional LSTM"""

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# BiLSTM Layer for sequence modeling
x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)  # LeakyReLU
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4, verbose=1)

history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

history_df = pd.DataFrame(history.history)
history_df.to_csv('modelLSTM_BI_LeakyReLU_.csv', index=False)
model.save('model_LSTM_BI_LeakyReLU.h5')

# Predictions on the test set
y_pred = model.predict(X_test_cnn)

# Evaluate the model
evaluate_model(y_test, y_pred, history=history)

"""**ELU**

Leaky ReLU is a variant of ReLU that introduces a small slope (α) for negative input values, thus preventing neurons from becoming completely inactive, which can happen with ReLU when negative values appear. ELU (Exponential Linear Unit) is an activation function that also aims to solve the problem of dead neurons, but it does this in a different way. ELU does not use a straight line for negative values, but instead uses an exponential transition that makes the negative output smoother and closer to zero.

LSTM
"""

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = ELU(alpha=0.01)(x)  # Using ELU activation
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# LSTM Layer for sequence modeling
x = LSTM(64, activation='tanh', return_sequences=True)(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = ELU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = ELU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4, verbose=1)

history = model.fit(X_train_cnn, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

history_df = pd.DataFrame(history.history)
history_df.to_csv('modelLSTM_ELU_.csv', index=False)
model.save('model_LSTM_ELU.h5')

# Predictions on the test set
y_pred = model.predict(X_test_cnn)

# Evaluate the model
evaluate_model(y_test, y_pred, history=history)

"""Bidirectional LSTM"""

# Define input shape for 1D sequential data
inputs = tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1))

# Feature extraction using 1D CNN
x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(inputs)
x = ELU(alpha=0.01)(x)  # Using ELU activation
x = GlobalAveragePooling1D()(x)  # Convert feature maps to a vector

# BiLSTM Layer for sequence modeling
x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(inputs)
x = GlobalAveragePooling1D()(x)  # Reduce LSTM output to meaningful features

# Fully Connected Layers
x = Dense(256, activation=None)(x)
x = ELU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = Dropout(0.19)(x)

x = Dense(32, activation=None)(x)
x = ELU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer for regression
outputs = Dense(1, activation='linear')(x)  # Linear activation for continuous output

# Create the model
model = Model(inputs, outputs)
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

history = model.fit(X_train_cnn, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

history_df = pd.DataFrame(history.history)
history_df.to_csv('modelLSTM_BI_ELU_complex.csv', index=False)
model.save('model_LSTM_BI_ELU_complex.h5')

# Predictions on the test set
y_pred = model.predict(X_test_cnn)

# Evaluate the model
evaluate_model(y_test, y_pred, history=history)

"""# **Comparison of training results of different models**

Load the training of all models
"""

LSTM_RELU = pd.read_csv("modelLSTM_Relu_.csv", sep=",")
LSTM_LEAKY = pd.read_csv("modelLSTM_LeakyReLU_.csv", sep=",")
LSTM_ELU = pd.read_csv("modelLSTM_ELU_.csv", sep=",")
BI_LSTM_RELU = pd.read_csv("modelLSTM_BI_Relu_.csv", sep=",")
BI_LSTM_LEAKY = pd.read_csv("modelLSTM_LeakyReLU_.csv", sep=",")
BI_LSTM_ELU = pd.read_csv("modelLSTM_BI_ELU_.csv", sep=",")

# Define differentiated colors
colors = {
    'LSTM_RELU': '#1E3A8A',     # Dark Blue (for LSTM)
    'LSTM_LEAKY': '#4F46E5',    # Light Blue (for LSTM)
    'LSTM_ELU': '#6B21A8',      # Purple (for LSTM)

    'BI_LSTM_RELU': '#EF4444',  # Red (for BI-LSTM)
    'BI_LSTM_LEAKY': '#F59E0B', # Orange Yellow (for BI-LSTM)
    'BI_LSTM_ELU': '#10B981'    # Green (for BI-LSTM)
}

# Plotting training results
import matplotlib.pyplot as plt

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(18, 7))

# Plot loss curves
ax[0].plot(LSTM_RELU['val_loss'], label="LSTM (ReLU)", color=colors['LSTM_RELU'])
ax[0].plot(LSTM_LEAKY['val_loss'], label="LSTM (LeakyReLU)", color=colors['LSTM_LEAKY'])
ax[0].plot(LSTM_ELU['val_loss'], label="LSTM (ELU)", color=colors['LSTM_ELU'])
ax[0].plot(BI_LSTM_RELU['val_loss'], label="BI-LSTM (ReLU)", color=colors['BI_LSTM_RELU'])
ax[0].plot(BI_LSTM_LEAKY['val_loss'], label="BI-LSTM (LeakyReLU)", color=colors['BI_LSTM_LEAKY'])
ax[0].plot(BI_LSTM_ELU['val_loss'], label="BI-LSTM (ELU)", color=colors['BI_LSTM_ELU'])
ax[0].set_title('Validation Loss Comparison')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot MAE (Mean Absolute Error) curves
ax[1].plot(LSTM_RELU['val_mae'], label="LSTM (ReLU)", color=colors['LSTM_RELU'])
ax[1].plot(LSTM_LEAKY['val_mae'], label="LSTM (LeakyReLU)", color=colors['LSTM_LEAKY'])
ax[1].plot(LSTM_ELU['val_mae'], label="LSTM (ELU)", color=colors['LSTM_ELU'])
ax[1].plot(BI_LSTM_RELU['val_mae'], label="BI-LSTM (ReLU)", color=colors['BI_LSTM_RELU'])
ax[1].plot(BI_LSTM_LEAKY['val_mae'], label="BI-LSTM (LeakyReLU)", color=colors['BI_LSTM_LEAKY'])
ax[1].plot(BI_LSTM_ELU['val_mae'], label="BI-LSTM (ELU)", color=colors['BI_LSTM_ELU'])
ax[1].set_title('Validation MAE Comparison')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('MAE')
ax[1].legend()

plt.tight_layout()
plt.show()

# You can also save the plot if you wish
fig.savefig("model_comparison_results.png")