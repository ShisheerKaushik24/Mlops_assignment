# Iris data Prediction and Analysis

This repository contains an end-to-end machine learning project on Iris data prediction. The project demonstrates an MLOps pipeline from data preprocessing to model deployment.

## Project Structure

```bash
project/
├── data/                          # Data files
├── src/
│   ├── data_loader.py             # Data loading script
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── feature_engineering.py     # Feature engineering
│   ├── train_model.py             # Model training and evaluation
│   ├── evaluate.py                # Model evaluation metrics
│   ├── pipeline.py                # ZenML pipeline
├── deployments/
│   ├── Dockerfile                 # Dockerfile for containerization
│   ├── app.py                     # Streamlit app for deployment
├── .github/
│   └── workflows/
│       └── mlflow_pipeline.yml              # GitHub Actions CI/CD configuration
├── README.md                      # Project documentation
├── requirements.txt               # Project dependencies
└── zenml_project/
    └── zenml.yaml                 # ZenML configuration
```
  
## Getting Started

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt

2. **Run ZenML Pipeline**:

   ```bash
   zenml pipeline run churn_prediction_pipeline

3. **Deploy**:

   ```bash
      docker build -t churn-app .
      docker run -p 8501:8501 churn-app

## Tools & Frameworks

   . **ZenML**: ML pipelines
   . **MLflow**: Experiment tracking
   . **Docker**: Containerization
   . **GitHub Actions**: CI/CD

## License
MIT License





