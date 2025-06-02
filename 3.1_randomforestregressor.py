# -*- coding: utf-8 -*-
"""3.1 RandomForestRegressor.ipynb

# **3. Creation of the Predictive Model**

Imports
"""

import pandas as pd
import numpy as np
from IPython.display import Image, display
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import warnings

"""# **3.1 RandomForestRegressor**

Why RandomForestRegressor? It's a regression model that uses multiple decision trees to make predictions. Each tree is trained on a random subset of data and features, which reduces their dependency and improves the model's performance. When a prediction is made, each tree gives its result, and the model averages those predictions.

This model is well suited for our case, as it avoids overfitting, can handle nonlinear relationships, and works well with noisy data or outliers. Additionally, it allows us to determine the importance of each feature for further modeling.

Load the processed dataset from .csv
"""

df_cleaned = pd.read_csv("dataframe_final.csv", sep=",")
df_cleaned.describe()

"""Importance of each independent variable with respect to the target variable"""

X = df_cleaned.drop(columns=['Tasa interanual del IPI', 'Valor'])

y = df_cleaned['Valor']  # Target variable

model = RandomForestRegressor()
model.fit(X, y)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Variable': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate MAE and MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print('Mean Absolute Percentage Error (MAPE): ', mae * 100 / (y_test.max() - y_test.min()))

"""**GridSearchCV**"""

# Random Forest model definition
rf_model = RandomForestRegressor(random_state=42)

# Reduce dataset size
from sklearn.utils import resample

# Subsample 30% of training data
X_train_small, y_train_small = resample(X_train, y_train, replace=False, n_samples=int(0.3 * len(X_train)), random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [300],
    'max_depth': [None, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'ccp_alpha': [0.001]
}

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1, verbose=2)
grid_search.fit(X_train_small, y_train_small)

# Best hyperparameters found
print("Best hyperparameters found:", grid_search.best_params_)

# Train model with best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_small, y_train_small)

# Predictions
y_pred = best_model.predict(X_test)

# Compute metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print('Mean Absolute Error (MAE) in %: ', mae * 100 / (y_test.max() - y_test.min()))

"""**Modification of some parameters**"""

from sklearn.preprocessing import StandardScaler

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize scaler
scaler = StandardScaler()

# Fit and transform X_train, only transform X_test
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ran_for_model = RandomForestRegressor(ccp_alpha=0.0005, max_depth=None, max_features=None,
                                      min_samples_leaf=2, min_samples_split=2, n_estimators=600)

# Train model
ran_for_model.fit(X_train, y_train)

# Predictions
y_pred = ran_for_model.predict(X_test)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Absolute Error (MAE) in %: ', mae * 100 / (y_test.max() - y_test.min()))
print(f"R² test: {r2:.4f}")

"""Plot prediction results vs actual values"""

plt.figure(figsize=(10, 4))
plt.scatter(y_test, y_pred, alpha=0.2, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')  # Ideal line
plt.title("Predictions vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.grid()
plt.show()

"""# **3.1.2 Quantifying Associated Uncertainty**

The uncertainty calculated using the standard deviation (std) of the predictions from each tree is a measure of the variability or dispersion of the trees' predictions in the model, for each target value — in our case, the energy demand.

*   68% of data falls within ±1 standard deviation
*   95% of data falls within ±2 standard deviations
*   99.7% of data falls within ±3 standard deviations
"""

warnings.filterwarnings('ignore')  # X has feature names, but DecisionTreeRegressor was fitted without feature names

# Get predictions from all trees
all_tree_preds = np.array([tree.predict(X_test) for tree in ran_for_model.estimators_])

# Mean of the predictions (final prediction)
final_prediction = np.mean(all_tree_preds, axis=0)

# Uncertainty (standard deviation of predictions)
prediction_uncertainty = np.std(all_tree_preds, axis=0)

# Create DataFrame
df_predictions = pd.DataFrame({
    'Prediction': final_prediction,
    'Uncertainty': prediction_uncertainty
})

# Show DataFrame
print(df_predictions.head())

print('Median uncertainty: ', df_predictions['Uncertainty'].median(), '/',
      'Relative to total: ', (df_predictions['Uncertainty'].median() / (y_test.max() - y_test.min())) * 100, '%')
print('Mean uncertainty: ', df_predictions['Uncertainty'].mean(), '/',
      'Relative to total: ', (df_predictions['Uncertainty'].mean() / (y_test.max() - y_test.min())) * 100, '%')

# Plot predictions vs uncertainty
plt.figure(figsize=(10, 3))
plt.scatter(final_prediction, prediction_uncertainty, alpha=0.07)
plt.title("Prediction vs Uncertainty")
plt.xlabel("Final Prediction")
plt.ylabel("Uncertainty (Standard Deviation)")
plt.show()

import seaborn as sns

def evaluate_model_mc(y_test, y_pred):
    # Calculate percentage errors
    Min = y_test.min()
    Max = y_test.max()
    Rang = Max - Min

    errors_percentual = (abs(y_test - y_pred) / abs(Rang)) * 100

    # Clip errors
    errors_percentual = np.clip(errors_percentual, 0, 30)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram with KDE
    sns.histplot(errors_percentual, kde=False, stat="percent", bins=60, ax=ax,
                 color="skyblue", edgecolor="black")

    # Chart aesthetics
    ax.set_title("Percentage Error Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Percentage Error (%)", fontsize=13)
    ax.set_ylabel("Sample Percentage (%)", fontsize=13)

    ax.set_xlim(left=0, right=5)
    ax.set_ylim(bottom=0)

    ax.set_xticks(np.arange(0, 5.1, 0.5))
    ax.set_xticklabels([f'{i}%' for i in np.arange(0, 5.1, 0.5)])

    ax.set_yticks(np.arange(0, 41, 5))
    ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 41, 5)])
    ax.grid(True)

    plt.tight_layout()
    plt.show()

evaluate_model_mc(y_test, y_pred)
