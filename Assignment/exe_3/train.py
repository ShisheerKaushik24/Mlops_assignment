import mlflow
import mlflow.sklearn
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set the experiment name
experiment_name = "model compare"
mlflow.set_experiment(experiment_name)

# Load the Titanic dataset from seaborn
data = sns.load_dataset('titanic')

# Data Preprocessing
mean_age = data['age'].mean()
mode_embarked = data['embarked'].mode()[0]
data['age'].fillna(mean_age, inplace=True)
data['embarked'] = data['embarked'].fillna(mode_embarked, inplace=True)
data['fare'].fillna(0, inplace=True)
data['deck'] = data['deck'].astype('category')
data['deck'] = data['deck'].cat.add_categories(['Unknown'])
data['deck'].fillna('Unknown', inplace=True)
mode_embarked_town = data['embark_town'].mode()[0]
data['embark_town'].fillna(mode_embarked_town, inplace=True)
data = data.drop_duplicates()

data['age'] = data['age'].astype('float')
data['fare'] = data['fare'].astype('float')  # Corrected typo 'astyoe'
data['sex'] = data['sex'].astype('category')
data['embarked'] = data['embarked'].astype('category')

# Feature Scaling
scaler = StandardScaler()
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])
data['family_size'] = data['sibsp'] + data['parch'] + 1

# Feature Selection
numerical_features = ['age', 'sibsp', 'parch', 'family_size']
categorical_features = ['sex', 'embarked', 'class', 'adult_male', 'deck', 'embark_town', 'pclass']

X = data[numerical_features + categorical_features]
y = data['fare']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Scaling the numerical features again after dummy variable conversion
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the best model
best_model = None
best_mse = float('inf')

# Create an example input for logging (keeping feature names by using DataFrame)
input_ex = X_train.iloc[[0]]  # Input example as DataFrame to retain feature names

# Train Linear Regression model
with mlflow.start_run() as run:
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_param("model_type", "Linear Regression")
    lr_predictions = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    mlflow.log_metric("mse", lr_mse)

    # Log the model
    mlflow.sklearn.log_model(lr_model, "linear_regression_model", input_example=input_ex)
    print("Linear Regression model logged.")

    if lr_mse < best_mse:
        best_mse = lr_mse
        best_model = lr_model

# Train Random Forest model
with mlflow.start_run() as run:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_param("model_type", "Random Forest")
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    mlflow.log_metric("mse", rf_mse)

    # Log the model
    mlflow.sklearn.log_model(rf_model, "random_forest_model", input_example=input_ex)
    print("Random Forest model logged.")

    if rf_mse < best_mse:
        best_mse = rf_mse
        best_model = rf_model

# Log the best model
if best_model is not None:
    mlflow.sklearn.log_model(best_model, "best_model", input_example=input_ex)
    print("Best model is logged.")
