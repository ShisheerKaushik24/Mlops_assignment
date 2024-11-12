import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier()
        }

    def run_experiment(self):
        mlflow.set_experiment("Iris Experiment Comparison")
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data()

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)

                mlflow.log_param("Model", model_name)
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Cross-Validation Mean", np.mean(cv_scores))
                mlflow.log_metric("Cross-Validation Std", np.std(cv_scores))