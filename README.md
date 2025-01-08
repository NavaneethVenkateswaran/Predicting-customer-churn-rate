# Predicting-customer-churn-rate

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import shap

# Load dataset
data = pd.read_csv('/mnt/data/telco_churn.csv')

# Inspect the dataset
print("Dataset Shape:", data.shape)
print("\nDataset Info:")
data.info()
print("\nMissing Values:")
print(data.isnull().sum())

# Univariate Analysis
print("\nUnivariate Analysis:")
for column in data.select_dtypes(include=['float64', 'int64']):
    print(f"{column} - Mean: {data[column].mean()}, Std: {data[column].std()}, Median: {data[column].median()}")

# Visualization for Univariate Analysis
for column in data.select_dtypes(include=['float64', 'int64']):
    sns.histplot(data[column], kde=True)
    plt.title(f"Histogram of {column}")
    plt.show()

# Bivariate Analysis
print("\nBivariate Analysis:")
if 'Churn' in data.columns:
    target = 'Churn'
    for column in data.select_dtypes(include=['float64', 'int64', 'object']):
        if column != target:
            print(f"Analyzing {column} vs {target}:")
            if data[column].dtype == 'object':
                sns.countplot(x=column, hue=target, data=data)
            else:
                sns.boxplot(x=target, y=column, data=data)
            plt.title(f"{column} vs {target}")
            plt.show()

# Additional Bivariate Analysis for specific pairs of variables
print("\nBivariate Analysis for Selected Pairs:")
pairs = [("MonthlyCharges", "TotalCharges"), ("tenure", "MonthlyCharges"), ("tenure", "TotalCharges")]
for var1, var2 in pairs:
    if var1 in data.columns and var2 in data.columns:
        sns.scatterplot(x=var1, y=var2, data=data)
        plt.title(f"Scatter Plot of {var1} vs {var2}")
        plt.show()

# Preprocessing for Feature Importance Analysis
if 'Churn' in data.columns:
    data[target] = data[target].map({'Yes': 1, 'No': 0})  # Adjust mapping as needed
X = data.drop(columns=[target])
y = data[target]
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature Importance Analysis with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)

# Hyperparameter Tuning with Random Search
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
random_search.fit(X_train, y_train)
print("Best Parameters from Random Search:", random_search.best_params_)

# Hyperparameter Tuning with Grid Search
grid_search = GridSearchCV(model, param_grid=param_dist, cv=3)
grid_search.fit(X_train, y_train)
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Final Model Evaluation
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

