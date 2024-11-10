import pickle
import pandas as pd
from feature_engineering import preprocess_features

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(data_path, model_path):
    data = pd.read_csv(data_path)
    data = preprocess_features(data)
    model = load_model(model_path)
    
    predictions = model.predict(data)
    return predictions

if __name__ == '__main__':
    predictions = predict('data/raw/churn_data.csv', 'models/churn_model.pkl')
    print(predictions)
