import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(data):
    # Example of handling categorical and numerical features
    categorical_features = ['Gender', 'Contract', 'PaymentMethod']
    numerical_features = ['MonthlyCharges', 'TotalCharges']

    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data
