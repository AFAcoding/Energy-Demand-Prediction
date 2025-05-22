# -*- coding: utf-8 -*-
"""3_1_2_XGBoost.ipynb

# **3.1.2 XGBoost** (Extreme Gradient Boosting )

XGBoost is a scalable and distributed gradient boosted decision tree (GBDT) machine learning library. It belongs to the family of ensemble learning models.

## **How does it work?**
**XGBoost** builds **sequential** decision trees. New trees are optimized to correct the errors of the previous ones. The term "gradient boosting" comes from the idea of "boosting" or improving a single weak model by combining it with a series of other weak models to create a strong collective model.

**GBDTs** iteratively train a set of shallow decision trees, with each iteration using the residual errors of the previous model to fit the next one. The final prediction is a weighted sum of all tree predictions. The GBDT boost minimizes bias and underfitting.

The gradient boosting method is enhanced by using the second derivative technique (or Taylor Expansion) on the loss function, which makes the model significantly faster and more effective compared to other traditional gradient boosting techniques.

Imports
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

"""Load .csv"""

df_cleaned = pd.read_csv("dataframe_final.csv",sep=",")
df_cleaned.describe()

# Convert categorical or object columns to numeric types
# For 'is_day' and 'Festiu', we assume they are binary indicators (0/1)
df_cleaned['is_day'] = df_cleaned['is_day'].map({'Yes': 1, 'No': 0})  # Adjust mapping as necessary
df_cleaned['Festiu'] = df_cleaned['Festiu'].map({'Yes': 1, 'No': 0})  # Adjust mapping as necessary

# For 'rain', ensure it's numeric; you may want to convert it directly
df_cleaned['rain'] = pd.to_numeric(df_cleaned['rain'], errors='coerce')

X = df_cleaned[['Any', 'Mes', 'Dia','Tram_Horari','Codi_Postal', 'temperature_2m',
                  'apparent_temperature','rain','wind_speed_10m','is_day',
                  'sunshine_duration','direct_radiation','Dia_Setmana',
                  'Festiu','dew_point_2m']]

y = df_cleaned['Valor']  # Variable objectiu

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mape=mae*100/(y_test.max()-y_test.min())
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

"""**Find best parameters**"""

xgb_model = XGBRegressor(objective='reg:squarederror')

# Define parameter grid
param_grid = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.15, 0.2, 0.25],
    'subsample': [0.9, 1.0, 1.1],
    'colsample_bytree': [0.6, 0.7,0.8]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

model = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.8, max_depth=6, learning_rate=0.25, random_state=42,reg_lambda=1, reg_alpha=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mape=mae*100/(y_test.max()-y_test.min())
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

"""**Exhaustive Feature Selection**"""

model = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.8, max_depth=6, learning_rate=0.25, random_state=42,reg_lambda=1, reg_alpha=0.5)

rfecv = RFECV(
    estimator=model,
    step=1,
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring='r2',
    min_features_to_select=7,
    verbose=2

)

rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)
print("Selected feature indices:", rfecv.get_support(indices=True))
print("Selected feature names:", X_train.columns[rfecv.get_support()])

X = df_cleaned[['Any', 'Mes', 'Dia', 'Tram_Horari', 'Codi_Postal', 'temperature_2m',
       'apparent_temperature', 'rain', 'wind_speed_10m', 'sunshine_duration',
       'direct_radiation', 'Dia_Setmana', 'Festiu', 'dew_point_2m']]

y = df_cleaned['Valor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.8, max_depth=6, learning_rate=0.25, random_state=42,reg_lambda=1, reg_alpha=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mape=mae*100/(y_test.max()-y_test.min())
r2 = r2_score(y_test, y_pred)

print(f"MAPE: {mape}")
print(f"R2: {r2}")

"""### **Method 1: XGBoost Prediction Interval using a Monte Carlo Ensemble**

It is a technique based on **random simulation** to estimate the uncertainty in a model's predictions. A single model is used, but variability is introduced into the prediction process.

This variability can be added through techniques such as **tree subsampling** in ensemble models. By making multiple predictions for the same input with small random variations, a **distribution of predictions** is obtained, which allows for the calculation of a prediction interval.

This technique is useful for obtaining a measure of uncertainty without needing to train multiple models, making it more computationally efficient.
"""

# Define a function to create a diverse Monte Carlo ensemble of XGBoost regressors
def diverse_monte_carlo_ensemble(X, y, n_models=200):
    models = []
    for i in range(n_models):
        # Fixed hyperparameters as per request
        model = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.7,
                             max_depth=6, learning_rate=0.2, random_state=i,reg_lambda=1, reg_alpha=0.5)
        model.fit(X, y)
        models.append(model)
    return models

# Create a diverse Monte Carlo ensemble of XGBoost regressors
ensemble = diverse_monte_carlo_ensemble(X_train, y_train)

# Make predictions with each model in the ensemble
predictions = np.column_stack([model.predict(X_test) for model in ensemble])

# Compute the 2.5th and 97.5th percentiles of the predictions for a 95% prediction interval
lower = np.percentile(predictions, 2.5, axis=1)
upper = np.percentile(predictions, 97.5, axis=1)

def evaluate_ensemble(y_true, predictions):
    # Compute the mean prediction across all models
    y_pred_mean = predictions.mean(axis=1)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_true, y_pred_mean)
    mape=mae*100/(y_test.max()-y_test.min())
    mse = mean_squared_error(y_true, y_pred_mean)
    r2 = r2_score(y_true, y_pred_mean)

    return {"MAE": mae,"MAE en %": mape, "MSE": mse, "R2 Score": r2}

# Evaluate the ensemble model
results = evaluate_ensemble(y_test, predictions)

# Print results
print("Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Visualize the true values vs. predictions with prediction intervals
plt.figure(figsize=(9, 5))
plt.scatter(y_test, predictions.mean(axis=1), alpha=0.5, label='Prediccions')
plt.errorbar(y_test, predictions.mean(axis=1), yerr=[predictions.mean(axis=1) - lower, upper - predictions.mean(axis=1)],
             fmt='none', ecolor='gray', alpha=0.3, label='95% Prediction Interval')
plt.plot(y_test, y_test, '--', color='red', label='Valors Reals')
plt.xlabel('Valors Reals')
plt.ylabel('Prediccions')
plt.title('Prediccions amb 95% dels intervals de predicció (Diverse Monte Carlo Ensemble)')
plt.legend()
plt.tight_layout()
plt.show()

"""### **Method 2: XGBoost Prediction Interval using a Bootstrap Ensemble**

It is a statistical technique based on the **creation of multiple random samples** from the available data. The goal is to obtain a better estimation of a prediction model’s uncertainty.

This technique relies on **sampling with replacement**, which means new data samples are created by randomly selecting elements from the original dataset, allowing repetitions.

Each of these samples is used to train an independent model. When making a prediction, all these models are used and their results are **combined to obtain a distribution of predictions**—a hybrid between XGBoost-style boosting and having **multiple trees** generating diverse regression values.

This distribution allows for estimating a prediction interval by capturing the range of variability in the generated predictions.

This technique is very useful when an estimate of uncertainty is needed **without making assumptions about the distribution of model errors**. Additionally, since it relies on multiple models, it can also help improve the generalization of predictions.
"""

# Define a function to create bootstrap samples and train XGBoost models
def bootstrap_models(X, y, n_bootstraps=100):
    models = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X.iloc[idx], y.iloc[idx]
        model = XGBRegressor(n_estimators=600, subsample=1.0, colsample_bytree=0.7,
                             max_depth=6, learning_rate=0.2, random_state=42,reg_lambda=1, reg_alpha=0.5)
        model.fit(X_boot, y_boot)
        models.append(model)
    return models

# Create an ensemble of XGBoost regressors using bootstrap aggregation
ensemble = bootstrap_models(X_train, y_train)

# Make predictions with each model in the ensemble
predictions = np.column_stack([model.predict(X_test) for model in ensemble])

# Compute the 2.5th and 97.5th percentiles of the predictions for a 95% prediction interval
lower = np.percentile(predictions, 2.5, axis=1)
upper = np.percentile(predictions, 97.5, axis=1)

def evaluate_ensemble(y_true, predictions):
    # Compute the mean prediction across all models
    y_pred_mean = predictions.mean(axis=1)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_true, y_pred_mean)
    mape=mae*100/(y_test.max()-y_test.min())
    mse = mean_squared_error(y_true, y_pred_mean)
    r2 = r2_score(y_true, y_pred_mean)

    return {"MAE": mae,"MAE en %": mape, "MSE": mse, "R2 Score": r2}

# Evaluate the ensemble model
results = evaluate_ensemble(y_test, predictions)

# Print results
print("Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Visualize the true values vs. predictions with prediction intervals
plt.figure(figsize=(9, 5))
plt.scatter(y_test, predictions.mean(axis=1), alpha=0.5, label='Prediccions')
plt.errorbar(y_test, predictions.mean(axis=1), yerr=[predictions.mean(axis=1) - lower, upper - predictions.mean(axis=1)],
             fmt='none', ecolor='gray', alpha=0.3, label='95% Interval de Predicció')
plt.plot(y_test, y_test, '--', color='red', label='Valors reals')
plt.xlabel('Valors reals')
plt.ylabel('Prediccions')
plt.title("Prediccions XGBoost amb l'interval del 95% de prediccions")
plt.legend()
plt.tight_layout()
plt.show()
