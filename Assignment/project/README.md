# Customer Churn Prediction and Analysis

This repository contains an end-to-end machine learning project on customer churn prediction. The project demonstrates an MLOps pipeline from data preprocessing to model deployment.

## Project Structure

- **data/**: Contains raw and processed data files.
- **notebooks/**: Jupyter notebooks for EDA, feature engineering, and model training.
- **src/**: Python scripts for data preprocessing, feature engineering, training, and prediction.
- **models/**: Trained model files.
- **mlruns/**: MLflow experiment tracking files.
- **docker/**: Dockerfile for containerizing the project.
- **.github/workflows/**: GitHub Actions workflow for CI/CD.

## Project Workflow

1. **Data Preprocessing**: Clean and split data, handle missing values, and scale features.
2. **Feature Engineering**: Apply transformations and encoding for categorical features.
3. **Model Training**: Train a RandomForestClassifier on the preprocessed data.
4. **Prediction**: Make predictions on new data using the trained model.

## Tools and Techniques

- **Environment Setup**: Docker for containerization, GitHub Actions for CI/CD
- **Feature Engineering**: Data transformation, one-hot encoding, scaling
- **Model Training**: RandomForestClassifier, model evaluation with accuracy score
- **Experiment Tracking**: MLflow for tracking experiments

## Getting Started

1. **Build Docker Image**
   ```bash
   docker build -t churn-prediction .
