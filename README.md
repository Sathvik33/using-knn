# Using KNN

This project demonstrates the use of the **K-Nearest Neighbors (KNN)** algorithm on the **Iris flower dataset**. It includes preprocessing, model training with various values of `K`, performance evaluation, and visualization of decision boundaries.

## 📝 Project Overview

The project explores the **K-Nearest Neighbors** algorithm, a simple yet powerful supervised machine learning method for classification. The **Iris dataset** is used to classify flowers into three species based on sepal and petal measurements. 

### Key Steps:
- **Data Preprocessing**: Includes label encoding for categorical variables and normalization for feature scaling.
- **Model Training**: Trains KNN models with different values of `K` to determine the best configuration.
- **Evaluation**: The model's performance is evaluated using key metrics such as accuracy, precision, recall, F1 score, and confusion matrix.
- **Visualization**: Visualizes the decision boundaries of the trained KNN model using two features (Sepal Length and Sepal Width).

## 🔍 Features

- **Multiple K Values**: Experiments with `K` values from 1 to 10.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Classification Report.
- **Visualization**: Plots the decision boundary of the KNN classifier for 2D data.

## 📊 Dataset

The **Iris dataset** consists of the following features:
- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**

The target variable is the **Species** of the flower, which includes three classes:
- Setosa
- Versicolor
- Virginica

## ⚙️ Libraries Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

