import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from  data_preprocessing import IrisDataProcessor
from train_model import IrisExperiment
from evaluate_model import IrisModelOptimizer


def main():
    processor = IrisDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data()

    experiment = IrisExperiment(processor)
    experiment.run_experiment()

    optimizer = IrisModelOptimizer(experiment)
    optimizer.quantize_model()
    optimizer.run_tests()

if __name__ == "__main__":
    main()