# -*- coding: utf-8 -*-
"""Copy of 3.1.2 XGBoost.ipynb

# **3. Predictive Model Creation**

Imports
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings

# Load the cleaned CSV dataset
df_cleaned = pd.read_csv("dataframe_final.csv", sep=",")

# Translate column names from Catalan/Spanish to English
column_translation = {
    'Any': 'Year',
    'Mes': 'Month',
    'Dia': 'Day',
    'Tram_Horari': 'Time_Slot',
    'Codi_Postal': 'Postal_Code',
    'Valor': 'Value',
    'temperature_2m': 'Temperature_2m',
    'apparent_temperature': 'Apparent_Temperature',
    'wind_speed_10m': 'Wind_Speed_10m',
    'sunshine_duration': 'Sunshine_Duration',
    'direct_radiation': 'Direct_Radiation',
    'Dia_Setmana': 'Weekday',
    'Tasa interanual del IPI': 'Yearly_IPI_Rate',
    'dew_point_2m': 'Dew_Point_2m'
}
df_cleaned.rename(columns=column_translation, inplace=True)

# Feature and target definition
X = df_cleaned[['Year', 'Month', 'Day', 'Time_Slot', 'Postal_Code',
                'Temperature_2m', 'Apparent_Temperature', 'rain', 'Wind_Speed_10m',
                'is_day', 'Sunshine_Duration', 'Direct_Radiation', 'Weekday', 'Festiu',
                'Yearly_IPI_Rate', 'Dew_Point_2m']]
y = df_cleaned['Value']

# Encode categorical values to binary (0 or 1)
df_cleaned['is_day'] = df_cleaned['is_day'].map({'Yes': 1, 'No': 0})
df_cleaned['Festiu'] = df_cleaned['Festiu'].map({'Yes': 1, 'No': 0})
df_cleaned['rain'] = pd.to_numeric(df_cleaned['rain'], errors='coerce')

# Drop some less important features
X = X.drop(columns=['Wind_Speed_10m','Temperature_2m','Dew_Point_2m','Yearly_IPI_Rate'])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
# **3.1.2 XGBoost** (Extreme Gradient Boosting)

Gradient boosting ensemble method with high performance.
"""

# Initial model
xgb_model = XGBRegressor(objective='reg:squarederror')

# Grid search to tune parameters
param_grid = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.15, 0.2, 0.25],
    'subsample': [0.9, 1.0, 1.1],
    'colsample_bytree': [0.6, 0.7, 0.8]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3,
                           scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Search for best number of estimators
model = XGBRegressor(subsample=1.0, colsample_bytree=0.7, max_depth=6,
                     learning_rate=0.2, random_state=42, objective="reg:squarederror",
                     reg_lambda=1, reg_alpha=0.5)
n_estimators = range(50, 650, 50)
param_grid = dict(n_estimators=n_estimators)
grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=4)
grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Plot estimator performance
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

plt.errorbar(n_estimators, means, yerr=stds)
plt.title("XGBoost n_estimators vs Log Loss")
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.show()

# Quantile Regression with XGBoost
quantiles = [0.05, 0.5, 0.95]
models = {}

# Train a model for each quantile
for q in quantiles:
    models[q] = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.7,
                             max_depth=6, learning_rate=0.2, random_state=42,
                             objective="reg:squarederror", reg_lambda=1, reg_alpha=0.5)
    models[q].fit(X_train, y_train)

# Predict each quantile
preds_05 = models[0.05].predict(X_test)
preds_50 = models[0.5].predict(X_test)
preds_95 = models[0.95].predict(X_test)

# Calculate bounds and median
lower_bound = preds_05
upper_bound = preds_95
median_prediction = preds_50

# Evaluate each model
print(f'MAE for 0.05 quantile: {mean_absolute_error(y_test, preds_05):.4f}')
print(f'MAE for 0.5 quantile: {mean_absolute_error(y_test, preds_50):.4f}')
print(f'MAE for 0.95 quantile: {mean_absolute_error(y_test, preds_95):.4f}')

# Print predictions
print("Lower bound (0.05 quantile):", lower_bound)
print("Median prediction (0.5 quantile):", median_prediction)
print("Upper bound (0.95 quantile):", upper_bound)

# Quantile loss evaluation
from sklearn.metrics import mean_pinball_loss

for quantile in quantiles:
    loss = mean_pinball_loss(y_test, models[quantile].predict(X_test), alpha=quantile)
    print(f"Quantile {quantile:.2f} Loss: {loss:.4f}")

# Evaluation of median model
y_pred_median = models[0.5].predict(X_test)
mae = mean_absolute_error(y_test, y_pred_median)
rmse = mean_squared_error(y_test, y_pred_median)
r2 = r2_score(y_test, y_pred_median)

print(f"Median Model Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted (median)
plt.figure(figsize=(9, 5))
plt.scatter(y_test, y_pred_median, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle="--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Median Values")
plt.title("Actual vs. Predicted (Median Model)")
plt.show()

"""
# **Method 2: Monte Carlo Ensemble for Prediction Intervals**
"""

# Create a diverse ensemble using random seeds
def diverse_monte_carlo_ensemble(X, y, n_models=200):
    models = []
    for i in range(n_models):
        model = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.7,
                             max_depth=6, learning_rate=0.2, random_state=i,
                             reg_lambda=1, reg_alpha=0.5)
        model.fit(X, y)
        models.append(model)
    return models

ensemble = diverse_monte_carlo_ensemble(X_train, y_train)

# Predict with all models
predictions = np.column_stack([model.predict(X_test) for model in ensemble])

# Compute 95% prediction interval
lower = np.percentile(predictions, 2.5, axis=1)
upper = np.percentile(predictions, 97.5, axis=1)

# Evaluate ensemble
def evaluate_ensemble(y_true, predictions):
    y_pred_mean = predictions.mean(axis=1)
    mae = mean_absolute_error(y_true, y_pred_mean)
    mse = mean_squared_error(y_true, y_pred_mean)
    r2 = r2_score(y_true, y_pred_mean)
    return {"MAE": mae, "MAE %": mae*100/(y_test.max()-y_test.min()), "MSE": mse, "R2 Score": r2}

results = evaluate_ensemble(y_test, predictions)

print("Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Visualize predictions and interval
plt.figure(figsize=(9, 5))
plt.scatter(y_test, predictions.mean(axis=1), alpha=0.5, label='Predictions')
plt.errorbar(y_test, predictions.mean(axis=1), yerr=[predictions.mean(axis=1) - lower, upper - predictions.mean(axis=1)],
             fmt='none', ecolor='gray', alpha=0.3, label='95% Prediction Interval')
plt.plot(y_test, y_test, '--', color='red', label='True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions with 95% Interval (Monte Carlo Ensemble)')
plt.legend()
plt.tight_layout()
plt.show()

"""
# **Method 3: Bootstrap Ensemble for Prediction Intervals**
"""

# Bootstrap method to estimate uncertainty
def bootstrap_models(X, y, n_bootstraps=100):
    models = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X.iloc[idx], y.iloc[idx]
        model = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.7,
                             max_depth=6, learning_rate=0.2, random_state=42,
                             reg_lambda=1, reg_alpha=0.5)
        model.fit(X_boot, y_boot)
        models.append(model)
    return models

ensemble = bootstrap_models(X_train, y_train)
predictions = np.column_stack([model.predict(X_test) for model in ensemble])
lower = np.percentile(predictions, 2.5, axis=1)
upper = np.percentile(predictions, 97.5, axis=1)

results = evaluate_ensemble(y_test, predictions)
print("Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Visualize results
plt.figure(figsize=(9, 5))
plt.scatter(y_test, predictions.mean(axis=1), alpha=0.5, label='Predictions')
plt.errorbar(y_test, predictions.mean(axis=1), yerr=[predictions.mean(axis=1) - lower, upper - predictions.mean(axis=1)],
             fmt='none', ecolor='gray', alpha=0.3, label='95% Prediction Interval')
plt.plot(y_test, y_test, '--', color='red', label='True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title("XGBoost Predictions with 95% Interval (Bootstrap Ensemble)")
plt.legend()
plt.tight_layout()
plt.show()
