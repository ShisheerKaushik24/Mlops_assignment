from zenml.pipelines import pipeline
from zenml.steps import step

@step
def data_loading_step():
    from src.data_loader import load_data
    return load_data()

@step
def data_preprocessing_step(data):
    from src.data_preprocessing import preprocess_data
    return preprocess_data(data)

@step
def feature_engineering_step(X_train, X_test):
    from src.feature_engineering import feature_engineering
    return feature_engineering(X_train, X_test)

@step
def model_training_step(X_train, y_train):
    from src.train_model import train_model
    return train_model(X_train, y_train)

@step
def evaluation_step(model, X_test, y_test):
    from src.evaluate import evaluate_model
    return evaluate_model(model, X_test, y_test)

@pipeline
def churn_prediction_pipeline(
    data_loading_step, 
    data_preprocessing_step, 
    feature_engineering_step, 
    model_training_step, 
    evaluation_step
):
    data = data_loading_step()
    X_train, X_test, y_train, y_test = data_preprocessing_step(data)
    X_train_scaled, X_test_scaled = feature_engineering_step(X_train, X_test)
    model = model_training_step(X_train_scaled, y_train)
    evaluation_step(model, X_test_scaled, y_test)
