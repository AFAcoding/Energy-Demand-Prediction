# TFG IN SMART AND SUSTAINABLE CITY MANAGEMENT, AUTONOMOUS UNIVERSITY OF BARCELONA (UAB) 
### Aleix Francía Albert

## Prediction of Energy Demand by Neighborhoods in Barcelona for Sustainable Resource Management

### Abstract
Electricity is a fundamental element of contemporary society and an indicator of technological development. The challenge of ensuring the sustainable use of resources is presented as a global priority, where optimization becomes an essential tool. According to the International Energy Agency (IEA), energy consumption in the European Union decreased by 3.2% in 2022, the second-largest drop since 2009. Although recovery is expected, negative electricity prices have become more common, reflecting market instability and the need for better management.

This study focuses on predicting electricity demand in the city of Barcelona, using artificial intelligence algorithms. These predictions and the associated uncertainty provide tools for the optimization and management of resources.

### Keywords
Energy efficiency, electricity consumption, demand forecasting, artificial intelligence, market volatility, Barcelona.

# INTRODUCTION

The growing demand for energy, combined with the need to reduce emissions of harmful gases, has made the efficient management of energy resources one of the major challenges in modern cities. This challenge is especially relevant as energy continues to be a limited resource, and due to the inability to store energy efficiently. It emphasizes the need to have a better understanding of the necessary production to reduce resource waste.

Global growth in energy consumption, particularly in regions like Asia and Oceania, where the Chinese economy has been one of the main drivers of energy consumption, has made research in energy efficiency (EE) increasingly important. Global energy consumption has increased by more than 50% between 1993 and 2012. As cities move toward more service-based economies, energy efficiency in production processes will not lose its relevance.

This study focuses on predicting electricity demand in Barcelona, broken down by postal codes. To address this challenge, various artificial intelligence and machine learning techniques, such as decision trees and neural networks, will be used, and the results of these will be measured through various metrics, with the aim of improving the accuracy and reliability of the predictions.

In addition to the creation of the predictive model, during this process, knowledge about the nature of the phenomenon and energy consumption in the Barcelona region can be extracted. This will enable a better understanding of local energy dynamics and improve the capacity to respond to fluctuations in demand.

One of the main objectives of this work is to generate a tool that allows for better management of energy resources and minimizes the impact of energy market volatility. Furthermore, the uncertainty associated with these predictions will be analyzed to ensure the reliability of the results obtained and provide an additional tool for decision-making.

In conclusion, the use of AI systems for energy demand forecasting represents a step towards more efficient management, making a more responsible use of available resources without compromising societal development and the needs associated with contemporary cities.

## 2. EDA (Exploratory Data Analysis)

Once the data preprocessing has been completed, an exploratory analysis will be developed using statistical and visual techniques to understand the nature of the energy demand phenomenon. The main objective is to identify patterns, trends, outliers, and relationships between variables, in order to gain a better understanding of the data and facilitate the subsequent development of the predictive model.

## 2.1 Distribution of Consumption by Postal Code

As can be seen, the distribution of energy consumption is not homogeneous across the surface of Barcelona, as there are significant differences in demand between the various neighborhoods of the city.

## 2.2 Histogram Analysis and Temporal Trend

The histogram and boxplot diagram show a trend towards the reduction of energy consumption over the years. This may be due to increased awareness of environmental impact and advancements in the energy efficiency of devices.

## 2.3 Monthly and Daily Consumption

Monthly consumption follows a pattern that influences energy consumption. However, we cannot observe the same type of influence with the distribution of daily consumption. The distribution of values depending on the day contains fluctuations but does not follow a clear trend.

## 2.4 Distribution by Time Slot

The time slot is one of the independent variables that directly contributes to demand, and this relationship is influenced by socioeconomic activity. The time slot from 12:00 PM to 6:00 PM has the highest consumption, while the slot with the least activity is from 12:00 AM to 6:00 AM, noted for its low variability. In all of these periods, where similar socioeconomic activity occurs, there is a similar standard deviation.

## 2.5 Influence of Independent Variables

To understand the influence of the different selected independent variables with respect to the dependent variable, the following heatmap has been created with different types of correlations.

- **Pearson**: Measures the linear relationship between two variables.
- **Spearman**: Measures the monotonic relationship (when two variables evolve in the same direction), but not necessarily in a linear way.
- **Distance Correlation (DCOR)**: Measures all types of dependence between two variables, whether linear or non-linear.

## 3. RANDOM FOREST REGRESSOR

The RandomForestRegressor is a regression model that uses a set of multiple decision trees to make predictions. This model uses a technique called **bagging** to generate complete decision trees in parallel, based on a random subset of the data and features. The final prediction is an average of all the predictions from the different decision trees.

It is suitable for our task because training each tree with different subsets of the data minimizes dependence between them, which increases its robustness and generalization capability. Furthermore, it is capable of handling non-linear relationships and works effectively with noisy data or outliers (atypical values).

## 3.1 Feature Selection

The Random Forest tool allows for calculating the importance of each variable, and along with the correlation matrix, it helps select the variables that provide the most information and better explain the behavior of the dependent variable. The results obtained with different sets of features indicate that the model fits the phenomenon better when a larger number of variables are used. However, to avoid overfitting, variables that do not contribute a significant value to the model are eliminated, as they may introduce noise.

Once the optimal features for modeling the phenomenon are selected, the **GridSearchCV** tool is used to identify the hyperparameters that provide the best performance for the dataset. This way, the model's accuracy is maximized without sacrificing its generalization.

## 3.2 Quantification of the Associated Uncertainty

With this model, we can also quantify the uncertainty associated with each prediction of the target variable. We calculate it using the standard deviation of the predictions from each of the trees.

## 4. EXTREME GRADIENT BOOSTING

XGBoost is a scalable and distributed decision tree learning technique augmented with gradient (GBDT), belonging to the ensemble learning model family. GBDT trains a set of shallow decision trees sequentially, with each iteration using the residual errors from the previous model. The new trees are optimized to correct the errors of the previous trees. The final prediction is a weighted sum of all the tree predictions.

Both Random Forest and GBDT construct a model consisting of multiple decision trees, but the difference lies in how the trees are built and combined. Random Forest minimizes variance and overfitting, while GBDT boosting minimizes bias and underfitting.

The term "gradient boosting" comes from the idea of boosting or improving a single weak model by combining it with a series of other weak models to generate a strong collective model. Gradient boosting sets objective results for the next model in an effort to minimize errors. The gradient boosting method is enhanced by using the second derivative technique (or Taylor Expansion) on the loss function, which makes the model much faster and more effective compared to traditional gradient boosting techniques.

## 4.1 Selected Hyperparameters

Hyperparameters control the learning process of a model. In the case of XGBoost, these hyperparameters determine the growth of the trees and the optimization of the model. A grid of hyperparameters was defined to explore different combinations of values and find the best configuration using an optimization technique called **GridSearchCV**, which has been used previously. Additionally, **Elastic Net** regularization was employed.

- **reg_lambda**: Applies a quadratic penalty to the weights of the trees, helping to reduce parameter variability and preventing some weights from becoming excessively large. The formula is:
  
  **L2 Regularization** = λ ∑(w_i^2)

  Where λ is the regularization coefficient and w_i are the tree weights.

- **reg_alpha**: Applies an absolute penalty to the weights, which encourages some features to become exactly zero, thus promoting sparsity in the model. The formula is:
  
  **L1 Regularization** = α ∑|w_i|

  Where α is the regularization coefficient and w_i are the tree weights.

- **Elastic Net**: Combines L1 (Lasso) and L2 (Ridge) penalties. The formula is:
  
  **Minimize Loss Function + λ1 ∑|w_i| + λ2 ∑(w_i^2)**

  Where λ1 controls the L1 penalty and λ2 controls the L2 penalty.
