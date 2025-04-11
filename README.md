# TFG IN SMART AND SUSTAINABLE CITY MANAGEMENT, AUTONOMOUS UNIVERSITY OF BARCELONA (UAB) 
### Aleix Francía Albert

## Prediction of Energy Demand by Neighborhoods in Barcelona for Sustainable Resource Management

### Abstract
Electricity is a fundamental element of contemporary society and an indicator of technological development. The challenge of ensuring the sustainable use of resources is presented as a global priority, where optimization becomes an essential tool. According to the International Energy Agency (IEA), energy consumption in the European Union decreased by 3.2% in 2022, the second-largest drop since 2009. Although recovery is expected, negative electricity prices have become more common, reflecting market instability and the need for better management.

This study focuses on predicting electricity demand in the city of Barcelona, using artificial intelligence algorithms. These predictions and the associated uncertainty provide tools for the optimization and management of resources.

### Keywords
Energy efficiency, electricity consumption, demand forecasting, artificial intelligence, market volatility, Barcelona.

# Introduction

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

## 3. Random Forest Regressor

The RandomForestRegressor is a regression model that uses a set of multiple decision trees to make predictions. This model uses a technique called **bagging** to generate complete decision trees in parallel, based on a random subset of the data and features. The final prediction is an average of all the predictions from the different decision trees.

It is suitable for our task because training each tree with different subsets of the data minimizes dependence between them, which increases its robustness and generalization capability. Furthermore, it is capable of handling non-linear relationships and works effectively with noisy data or outliers (atypical values).

## 3.1 Feature Selection

The Random Forest tool allows for calculating the importance of each variable, and along with the correlation matrix, it helps select the variables that provide the most information and better explain the behavior of the dependent variable. The results obtained with different sets of features indicate that the model fits the phenomenon better when a larger number of variables are used. However, to avoid overfitting, variables that do not contribute a significant value to the model are eliminated, as they may introduce noise.

Once the optimal features for modeling the phenomenon are selected, the **GridSearchCV** tool is used to identify the hyperparameters that provide the best performance for the dataset. This way, the model's accuracy is maximized without sacrificing its generalization.

## 3.2 Quantification of the Associated Uncertainty

With this model, we can also quantify the uncertainty associated with each prediction of the target variable. We calculate it using the standard deviation of the predictions from each of the trees.

## 4. Extreme Gradient Boosting

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

## 4.2 Monte Carlo Ensemble

It is a technique based on random simulation to estimate the uncertainty in a model's predictions. Instead of training multiple models like in **Bootstrap Ensemble**, a single model is used, but variability is introduced in the prediction process. This variability can be added through techniques such as subsampling trees in ensemble models. By making multiple predictions for the same input but with small random variations, a distribution of predictions is obtained, which allows for the calculation of a prediction interval in a manner similar to what is done in **Bootstrap Ensemble**.

This technique is widely used when one wants to obtain a measure of uncertainty without having to train multiple models, making it more efficient in terms of computation.

## 4.3 Bootstrap Ensemble

It is a statistical technique based on creating multiple random samples from the available data. The goal is to obtain a better estimate of the uncertainty of a prediction model. This technique is based on bootstrapping, meaning that new data samples are created by randomly selecting elements from the original sample, allowing repetitions.

Each of these samples is used to train an independent model. When we want to make a prediction, all these models are used, and their results are combined to obtain a distribution of predictions. It is a combination of using the boost from XGBoost while also having multiple trees that generate different regression values. This distribution allows for estimating a prediction interval by capturing the range of variability of the generated predictions.

This technique is very useful when one wants to have an estimate of uncertainty without needing to make assumptions about the distribution of the model's errors. Additionally, since it relies on the use of multiple models, it can also help improve the generalization of the predictions.

## 5 Neural Network (CNN)

Next, we will conduct several tests on an initial model. The model is as follows:

## 5.1 Long Short-Term Memory (LSTM)

First, the model was trained and evaluated using the ReLU activation and introducing the concept of LSTM (Long Short-Term Memory), which is a type of RNN (Recurrent Neural Network) designed to capture long-term dependencies in data sequences. This is possible due to the structure of gates. The gate structure allows for the retention of relevant information over long periods, whereas traditional RNNs cannot do this.

LSTMs operate through the following gates that control the flow of information:

- **Forget Gate**: Decides which information from the memory should be retained or discarded. The gate value is between 0 (information is discarded) and 1 (information is retained).
  
- **Input Gate**: Decides which new information is added to the memory, combining the current input with the previous state. The input gate uses an activation function (usually a sigmoid) to control the update.
  
  $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

- **Output Gate**: Determines which information is transmitted to the next layer or as output. It also uses a sigmoid activation to decide the flow of information.

The traditional LSTM processes the input sequence in a single direction (usually left to right). The state of the network at each moment depends on the input and the previous state, learning past dependencies to predict future ones.

## 5.2 Activation

Activation is the process through which a neuron transforms its input (a weighted value) into an output using an activation function. It is fundamental for introducing non-linearity into the neural network, which is the essence of neural networks. For our study, we will use various variations of the ReLU function, which is the activation function initially implemented in the model.

1. **ReLU (Rectified Linear Unit)**:
   The ReLU function outputs the input directly if it is positive, otherwise, it outputs zero.

   $$ f(x) = \max(0, x) $$

   - If \( x > 0 \), then \( f(x) = x \).
   - If \( x \leq 0 \), then \( f(x) = 0 \).

   **Example**:
   - If \( x = 5 \), then \( f(x) = \max(0, 5) = 5 \).
   - If \( x = -3 \), then \( f(x) = \max(0, -3) = 0 \).

---

2. **Leaky ReLU (Leaky Rectified Linear Unit)**:
   Leaky ReLU is similar to ReLU, but for negative values of \( x \), instead of returning zero, it returns a small negative value determined by \( \alpha \), which is a small constant.

   $$ f(x) = 
   \begin{cases}
     x & \text{if } x > 0 \\
     \alpha x & \text{if } x < 0
   \end{cases} $$

   - If \( x > 0 \), then \( f(x) = x \) (same as ReLU).
   - If \( x \leq 0 \), then \( f(x) = \alpha x \), where \( \alpha \) is a small constant (usually between 0 and 1).

   **Example** (Assume \( \alpha = 0.01 \)):
   - If \( x = 5 \), then \( f(x) = 5 \) (since \( x > 0 \)).
   - If \( x = -3 \), then \( f(x) = 0.01 \times (-3) = -0.03 \).

---

3. **ELU (Exponential Linear Unit)**:
   ELU works similarly to ReLU for positive values of \( x \), but for negative values, it uses an exponential function to avoid the output being zero and allows for negative outputs.

   $$ f(x) = 
   \begin{cases}
     x & \text{if } x > 0 \\
     \alpha (e^x - 1) & \text{if } x < 0
   \end{cases} $$

   - If \( x > 0 \), then \( f(x) = x \) (same as ReLU).
   - If \( x \leq 0 \), then \( f(x) = \alpha (e^x - 1) \), where \( \alpha \) is a constant (typically 1).

   **Example** (Assume \( \alpha = 1 \)):
   - If \( x = 5 \), then \( f(x) = 5 \) (since \( x > 0 \)).
   - If \( x = -3 \), then \( f(x) = 1 \times (e^{-3} - 1) \approx 1 \times (0.0498 - 1) = -0.9502 \).

---
As seen in the following table, developed from the study [1], each activation function behaves differently depending on the type of dataset. Significant differences can be observed between them, both in computation time and the results.

## 5.3 Bidirectional LSTM

The bidirectional LSTM consists of two parallel input paths. One processes the sequence from left to right, while the other does so from right to left. This feature allows the model to process the data sequence in both directions, providing the network with more context of the sequence.

To evaluate the different types of models, we will generate a model for each activation type mentioned, both with traditional LSTM and with bidirectional LSTM, all of them initially structured in the same way.

## 5.4 Evaluation Metrics

The metrics used during the study to evaluate how suitable the model and the chosen activation type are for our task will be:

- **MSE (Mean Squared Error)**: The average of the squared differences between the predicted and actual values. It penalizes predictions that are farther away.
  $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

- **MAPE (Mean Absolute Percentage Error)**: The average of the absolute relative errors in percentage.
  $$ MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100 $$

- **R² (Coefficient of Determination)**: Indicates the proportion of variability explained by the model. A value close to 1 indicates a good fit.
  $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$

## 5.5 Results

As we can observe, the models where the bidirectional LSTM was applied generally obtained better results using the same activation function. However, the best result was achieved through the traditional LSTM model with the ELU activation function.

On the other hand, it can be affirmed that the ELU activation function, which is the most computationally expensive, has outperformed the other activation functions from the ReLU family, followed by LeakyReLU, and lastly ReLU, which is the least costly among them. 

For the test data, we can observe that all models achieved better results than during training, but with very similar metrics to those of the validation set. This phenomenon is caused by the use of strong regularization, in our case, the Dropout layer and Batch Normalization.

## 5.6 Effect of Batch Normalization

Batch Normalization normalizes the activations of each layer using the mean and variance of the batch during training. However, during inference (validation and testing), it uses the accumulated mean and variance from the entire training set, making the predictions more stable.

- **During training**: The layer normalizes its output using the mean and standard deviation of the current batch of inputs. That is, for each channel being normalized, it helps with convergence. The layer returns:
  $$ \gamma \cdot \frac{batch - \text{mean(batch)}}{\sqrt{\text{var(batch)} + \epsilon}} + \beta $$

  The terms are explained as follows:
  - **ϵ** is a small constant (configurable as part of the constructor arguments).
  - **γ** is a learned scaling factor (initialized as 1), which can be disabled by passing `scale=False` to the constructor.
  - **β** is a learned offset factor (initialized as 0), which can be disabled by passing `center=False` to the constructor.

- **During validation/test**: The layer normalizes its output using a moving average of the mean and standard deviation of the batches seen during training, making the model more stable and achieving better generalization. It returns:
  $$ \gamma \cdot \frac{batch - \text{self.moving mean}}{\sqrt{\text{self.moving var} + \epsilon}} + \beta $$

  The terms are explained as follows:
  - **self.moving mean** and **self.moving var** are non-trainable variables that are updated every time the layer is called in training mode, as follows:
    $$ \text{moving mean} = \text{moving mean} \cdot \text{momentum} + \text{mean(batch)} \cdot (1 - \text{momentum}) $$
    $$ \text{moving var} = \text{moving var} \cdot \text{momentum} + \text{var(batch)} \cdot (1 - \text{momentum}) $$

  The layer will only normalize its inputs during inference after having been trained with data that has statistics similar to the inference data.

## 5.7 Effect of Dropout

Dropout is a regularization technique that aims to randomly deactivate a percentage of neurons during each forward pass of training. This forces the model to be more robust and prevents excessive dependence on certain neurons.

- **During training**: The model trains with a "degraded" architecture, reducing its capacity and increasing loss during the process.
  
- **During validation/test**: The Dropout regularization technique is not applied, and all neurons are active, allowing the model to function at its full potential.

## 6 Conclusions

In conclusion, we have tested and evaluated several machine learning models, including Random Forest, XGBoost, and LSTM networks, specifically focusing on how different configurations and techniques impact performance.

- **Random Forest and XGBoost**: Both models showed excellent performance with high generalization capabilities, especially when using feature selection methods and tuning hyperparameters with GridSearchCV. The ability of Random Forest to handle high-dimensional datasets and the sequential correction of errors in XGBoost with gradient boosting proved to be highly effective.

- **LSTM Networks**: The LSTM models, particularly when using bidirectional LSTM layers, were effective in capturing long-term dependencies in sequential data. Among the various activation functions tested, ELU showed superior results in terms of prediction accuracy, despite being computationally more expensive. The addition of Dropout and Batch Normalization improved model stability and prevented overfitting.

- **Model Evaluation**: The evaluation metrics, such as MSE, MAPE, and R², helped in comparing the different models' performances. The models with LSTM (especially bidirectional) generally outperformed others, and the most significant improvement was observed when using the ELU activation function. The dropout and batch normalization techniques further enhanced the generalization of the models.

In summary, **LSTM with ELU activation** and **XGBoost** models offered the best overall performance for the task. However, the choice of the model and its configuration depends heavily on the specific problem and dataset at hand. Future work could involve experimenting with different neural network architectures, regularization techniques, and further fine-tuning of hyperparameters to improve the models' performance.
