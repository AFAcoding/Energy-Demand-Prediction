# **Prediction of Energy Demand by Neighborhood in Barcelona for Sustainable Resource Management**

## Overview
This repository contains the code and analysis for predicting **electricity demand** in **Barcelona**, disaggregated by **postal code**, using **machine learning algorithms**. The project was developed as part of a **thesis** and includes uncertainty quantification techniques to ensure reliable predictions. The models are trained on **multidimensional datasets** that include electricity consumption, meteorological data, and economic indicators.

The goal of the project is to provide insights into the factors influencing energy consumption and help optimize resource management for a more **sustainable** and **efficient** energy system in urban environments.

## Key Features
- **Prediction Models**: The code includes implementations of multiple machine learning algorithms, including:
  - **Random Forest Regressor**
  - **Extreme Gradient Boosting (XGBoost)**
  - **Neural Networks (LSTM and Bidirectional LSTM)**

- **Uncertainty Quantification**: The uncertainty associated with predictions is calculated for all models using techniques such as:
  - **Monte Carlo Dropout**
  - **Ensemble Methods**

- **Multidimensional Dataset**: Data includes electricity consumption, weather information, and economic indicators, covering the years **2019-2024** for Barcelona.
  - **Electricity Consumption Data**: Sourced from the **Barcelona Energy Observatory** and the **Barcelona Open Data portal**.
  - **Meteorological Data**: Sourced from the **Open Meteo API**.
  - **Economic Indicators**: Sourced from the **National Statistics Institute (INE)**.

- **Data Processing and Feature Engineering**: The repository includes scripts for cleaning, preprocessing, and feature engineering to prepare the dataset for model training.

# Project Structure

```
Barcelona-Energy-Demand
│
├── data/
│ └── raw_data/ # Raw datasets
│ └── processed_data/ # Preprocessed data
│
├── models/
│ └── random_forest.py # Random Forest model
│ └── xgboost.py # XGBoost model
│ └── lstm.py # LSTM model
│ └── uncertainty.py # Uncertainty estimation methods
│
├── notebooks/ # Jupyter notebooks for exploratory analysis
│ └── EDA.ipynb # Exploratory Data Analysis
│
├── requirements.txt # List of dependencies
├── train_models.py # Script to train models
├── evaluate_models.py # Script to evaluate models
├── preprocess_data.py # Data preprocessing script
└── uncertainty_estimation.py # Script for uncertainty estimation
```

# Results

Here are the final results from the models' performance:

### LSTM Models

| **Model**                        | **MAPE** | **R²**   |
|----------------------------------|----------|----------|
| LSTM ELU                         | 1.7424   | **0.9683** |
| LSTM ELU + Monte Carlo Dropout  | 1.7648   | 0.9678   |
| LSTM LEAKY                       | 1.8567   | 0.9647   |
| LSTM RELU                        | 2.0296   | 0.9591   |

### Bidirectional LSTM Models

| **Model**       | **MAPE** | **R²**   |
|-----------------|----------|----------|
| BI LSTM ELU     | 1.7919   | **0.9671** |
| BI LSTM LEAKY   | 1.8283   | 0.9659   |
| BI LSTM RELU    | 2.2039   | 0.9550   |

### XGBoost Models

| **Model**                          | **MAPE** | **R²**   |
|-----------------------------------|----------|----------|
| XGBoost with Monte Carlo Ensemble | 1.7417   | **0.9665** |
| XGBoost with Bootstrap Ensemble   | 1.8020   | 0.9647   |

### Random Forest Model

| **Model**              | **MAPE** | **R²**   |
|------------------------|----------|----------|
| RandomForestRegressor  | 1.4533   | 0.9684   |

# Contact

**Aleix Francia Albert**  
**Email**: afranciaa2501@gmail.com  
**Thesis supervised by**: Albert Romero Sánchez (Department of Computer Science)

