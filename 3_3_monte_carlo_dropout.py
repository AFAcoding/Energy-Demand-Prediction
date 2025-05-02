# -*- coding: utf-8 -*-
"""3.3 Monte Carlo Dropout.ipynb

### **Monte Carlo Dropout (MC Dropout)**

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
from tensorflow.keras.layers import Conv1D, Layer, MaxPooling1D, Dropout, Dense, GlobalAveragePooling1D, BatchNormalization, LSTM, LeakyReLU, ELU, Bidirectional, Multiply, Permute, Reshape, RepeatVector, Lambda, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam

"""Load Data"""

df_cleaned = pd.read_csv("dataframe_final.csv", sep=",")

X_reduced = df_cleaned.drop(columns=['Year-on-Year IPI Rate', 'Value'])
y = df_cleaned['Value']  # Target variable

"""Use the same data normalization method"""

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y = np.array(y)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Transform data for CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Add channel, (N samples, features) -> (N samples, features, dimension)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

"""Model"""

LSTM_ELU_model = tf.keras.models.load_model("model_LSTM_ELU.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

def evaluate_model(y_test, y_pred):
    # Calculate basic metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'R²: {r2:.4f}')
    # Calculate the range of actual data
    Min = y_test.min()
    Max = y_test.max()
    Range = Max - Min

    mape = (mae * 100) / Range
    print(f'MAPE in %: {mape:.4f}%')

# Predictions on the test set
y_pred = LSTM_ELU_model.predict(X_test_cnn)

# Evaluate the model
evaluate_model(y_test, y_pred)

"""# **Performing Monte Carlo Dropout Inference.**"""

def monte_carlo_predictions(model, X, n_simulations):
    f_preds = []
    for _ in range(n_simulations):
        preds = model(X, training=True)
        f_preds.append(preds.numpy())

    f_preds = np.array(f_preds)

    mean_preds = np.mean(f_preds, axis=0)
    std_preds = np.std(f_preds, axis=0)

    return mean_preds, std_preds

mean_preds, std_preds = monte_carlo_predictions(LSTM_ELU_model, X_test, 100)

"""Model Evaluation"""

def evaluate_model_mc(y_test, y_pred_mean, y_pred_std):
    # Flatten the arrays
    y_test = y_test.flatten()
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()

    # Basic metrics
    mae = mean_absolute_error(y_test, y_pred_mean)
    mse = mean_squared_error(y_test, y_pred_mean)
    r2 = r2_score(y_test, y_pred_mean)

    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')

    # Metrics based on the interval
    Min = y_test.min()
    Max = y_test.max()
    Range = Max - Min

    mape = (mae * 100) / Range
    print(f'MAPE in %: {mape:.4f}%')

    # --- Plots ---
    plt.figure(figsize=(9, 5))

    # 1. Visualize actual values vs. predictions with prediction intervals
    plt.scatter(y_test, y_pred_mean, alpha=0.5, label='Predictions')
    lower = y_pred_mean - 2 * y_pred_std
    upper = y_pred_mean + 2 * y_pred_std

    # Error bars representing the prediction interval
    plt.errorbar(y_test, y_pred_mean, yerr=[y_pred_mean - lower, upper - y_pred_mean], fmt='none', ecolor='gray', alpha=0.1, label='95% Prediction Interval')

    # Continuous red line representing perfect prediction (where actual values are equal to predictions)
    plt.plot(y_test, y_test, '-', color='red', label='Actual Values')

    # Labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title("Predictions with 95% Prediction Interval")
    plt.legend()
    plt.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

evaluate_model_mc(y_test, mean_preds, std_preds)

def evaluate_model_mc(y_test, y_pred_mean, y_pred_std):
    # Flatten arrays
    y_test = y_test.flatten()
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()

    # Calculate percentage errors
    Min = y_test.min()
    Max = y_test.max()
    Range = Max - Min

    percentual_errors = (abs(y_test - y_pred_mean) / abs(Range)) * 100

    # Clip the percentage errors to a maximum value
    percentual_errors = np.clip(percentual_errors, 0, 30)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Histogram with KDE
    sns.histplot(percentual_errors, kde=True, stat="density", bins=60, ax=ax, color="skyblue", edgecolor="black")

    # Aesthetics of the plot with larger, bold titles
    ax.set_title("Distribution of Percentage Errors", fontsize=14, fontweight='bold')
    ax.set_xlabel("Percentage Error (%)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)

    ax.set_xlim(left=0, right=5)
    ax.set_ylim(bottom=0)

    # Add more labels to the X-axis
    ax.set_xticks(np.arange(0, 5.1, 0.5))
    ax.set_xticklabels([f'{i}%' for i in np.arange(0, 5.1, 0.5)])

    # Add labels to the Y-axis from 0.1 to 0.1
    ax.set_yticks(np.arange(0, 0.7, 0.1))
    ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 0.7, 0.1)])

    # Add grid
    ax.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

evaluate_model_mc(y_test, mean_preds, std_preds)
