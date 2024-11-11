# Quantized Machine Learning Experiments

This repository contains two scripts to explore the effects of quantization in machine learning:

1. **`quantized_classification.py`**: Uses different precision levels to quantize the input data for various classifiers (Decision Tree, k-NN, and SVM) on the Wine dataset.
2. **`quantized_sgd_classifier.py`**: Implements a custom SGD classifier with quantized gradients on the Breast Cancer dataset.

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```
## Usage

    1. **For Quantized Classification**:
    ```bash
    python quantized_classification.py
    ```

    2. **For Quantized SGD Classifier**:
    ```bash
    python quantized_sgd_classifier.py
    ```
