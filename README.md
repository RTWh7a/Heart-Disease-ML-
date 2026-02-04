# Heart-Disease-ML-
Heart Disease Classification & Analysis 🩺
This project implements a comprehensive machine learning pipeline to predict the presence of heart disease using clinical data. It evaluates multiple classification algorithms, applies dimensionality reduction, and explores data structures through unsupervised clustering.

📊 Project Overview
The primary goal is to compare various machine learning models to determine which algorithm best predicts heart disease based on patient attributes (age, sex, chest pain type, cholesterol levels, etc.).

🛠️ Key Features
Exploratory Data Analysis (EDA): Visualizes data distribution using Boxplots, Scatter plots, Pie charts, and Count plots to understand gender distribution and heart disease prevalence.

Preprocessing: Includes handling missing values, standardizing features using StandardScaler, and splitting data into training and testing sets.

Predictive Modeling: Implements and evaluates six different classifiers:

Logistic Regression

Naive Bayes

Random Forest

K-Nearest Neighbors

Decision Tree (with visualization)

Support Vector Machine (SVM)

Performance Metrics: Detailed evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices for every model.

Unsupervised Learning:

PCA (Principal Component Analysis): Reduced feature dimensionality while retaining 95% variance.

K-Means & Hierarchical Clustering: Explores natural groupings in the data through centroids and dendrograms.

Model Diagnostics: Generates Learning Curves to detect overfitting/underfitting and Feature Importance plots to identify which clinical factors drive the predictions.

🚀 Technologies Used
Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Scipy

📈 Results Summary
The script automatically identifies and prints the best-performing model based on testing accuracy, ensuring the most reliable predictor is highlighted for the specific dataset.
