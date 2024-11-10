import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from feature_engineering import preprocess_features
from data_preprocessing import load_data, split_data, save_data

def train_model(data_path):
    # Load and preprocess data
    data = load_data(data_path)
    data = preprocess_features(data)
    train, test = split_data(data)
    
    X_train = train.drop('Churn', axis=1)
    y_train = train['Churn']
    X_test = test.drop('Churn', axis=1)
    y_test = test['Churn']

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate and save model
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'Model Accuracy: {accuracy:.2f}')
    
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    train_model('data/raw/churn_data.csv')
