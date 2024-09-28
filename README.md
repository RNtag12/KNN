
# K-Nearest Neighbors (KNN) Classifier Implementation and Optimization
This project explains how to implement and optimize a K-Nearest Neighbors (KNN) classifier using Python. The project involves data preprocessing, visualization, model training, and performance evaluation to achieve an optimal K value for the classifier.

## General Information
Machine learning (ML) is a subfield of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to perform specific tasks without explicit instructions. One popular ML algorithm is the K-Nearest Neighbors (KNN) classifier, which is used for both classification and regression tasks. It works by finding the 'k' nearest data points in the training set to a new data point and making predictions based on the majority class among these neighbors.

KNN is simple yet powerful, making it a widely used algorithm for various applications, from medical diagnosis to financial forecasting. However, it is essential to scale features before applying KNN because it is a distance-based algorithm.

The official documentation for the Scikit-learn library, which we use in this project, can be found at https://scikit-learn.org/stable/.

## Project Description
This project demonstrates the implementation and optimization of a KNN classifier using the Scikit-learn library in Python. The dataset used for this project is preprocessed to ensure that all features are scaled properly. Visualization techniques are employed to understand the data distribution and relationships between features. The project aims to find the optimal number of neighbors ('k') for the KNN classifier to achieve the best performance.

## Features
Data Loading and Preprocessing: Reading the dataset, handling missing values, and scaling features.
Data Visualization: Using Seaborn to create pair plots to visualize the relationships between features and the target class.
Model Training and Evaluation: Implementing KNN, training the model, and evaluating its performance using confusion matrices and classification reports.
Hyperparameter Optimization: Finding the optimal 'k' value by plotting error rates for different 'k' values.
## How It Works

- <b> Data Loading and Preprocessing</b>
  
The dataset is loaded using Pandas, and the features are scaled using the StandardScaler to ensure that all features contribute equally to the distance calculation in KNN.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('KNN Dataset', index_col=0)
data.head()
```

- <b>Visualization of data</b>

```python
sns.pairplot(data, hue='TARGET CLASS', palette='coolwarm')
```

- <b>Scaling the features</b>

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(data.drop('TARGET CLASS', axis=1))
data_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])
```
- <b>Model Training and Evaluation</b>

The dataset is split into training and testing sets. A KNN classifier is trained on the training data, and its performance is evaluated on the testing data.

```python
from sklearn.model_selection import train_test_split
X = data_feat
Y = data['TARGET CLASS']
X_train, X_test, Y_train, Y_test = train_test_split(scaled_features, data['TARGET CLASS'], test_size=0.30)
```
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)
```
```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))
```
- <b>Hyperparameter Optimization</b>

The optimal 'k' value is found by testing different values and plotting the error rate for each. The optimal value minimizes the error rate.
```python
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', markerfacecolor='red', marker='o', markersize=10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```
- <b>Final Model Evaluation</b>

The KNN classifier is retrained using the optimal 'k' value, and its performance is evaluated.

```python
knn = KNeighborsClassifier(n_neighbors=34)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)

print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))
```

- <b>Steps to Execute the Project</b>
    - <b>Clone the repository or download the script</b>.
    - <b>Install the required dependencies</b>
         ```python
        pip install numpy pandas seaborn matplotlib scikit-learn
         ```
     - <b> Ensure the dataset is in the same directory as the script or provide the correct path to the dataset then run the script. </b>

        ```python
        python knn_classifier_optimization.py
        ```

## Conclusion
By implementing and optimizing a KNN classifier, this project showcases the importance of feature scaling and hyperparameter tuning in machine learning. The error rate plot helps identify the optimal number of neighbors, significantly improving the classifier's accuracy from 0.75 to 0.87. This method can be applied to various datasets to achieve reliable and accurate predictions.
