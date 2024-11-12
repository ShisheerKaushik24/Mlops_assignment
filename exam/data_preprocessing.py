import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class IrisDataProcessor:
    def __init__(self):
        self.data = load_iris()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        self.df = pd.DataFrame(
            data=np.c_[self.data['data'], self.data['target']],
            columns=self.data['feature_names'] + ['target']
        )
        scaler = StandardScaler()
        features = self.df[self.data['feature_names']]
        self.df[self.data['feature_names']] = scaler.fit_transform(features)
        X = self.df[self.data['feature_names']]
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test