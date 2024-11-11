# quantized_classification.py
"""
This script evaluates the effect of quantizing input data on the accuracy of various classifiers
(Decision Tree, k-NN, and SVM) using the Wine dataset.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = load_wine().data
target = load_wine().target

def quantize_data(data, bits):
    levels = 2 ** bits
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    quantized_data = np.round(((data - min_val) / (max_val - min_val)) * (levels - 1)) * (max_val - min_val) / (levels - 1) + min_val
    return quantized_data

data_8bit = quantize_data(data, 8)
data_4bit = quantize_data(data, 4)
data_2bit = quantize_data(data, 2)

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC()
}

accuracy_full = {}
accuracy_8bit = {}
accuracy_4bit = {}
accuracy_2bit = {}

for name, model in models.items():
    accuracy_full[name] = cross_val_score(model, data, target, cv=5).mean()
    accuracy_8bit[name] = cross_val_score(model, data_8bit, target, cv=5).mean()
    accuracy_4bit[name] = cross_val_score(model, data_4bit, target, cv=5).mean()
    accuracy_2bit[name] = cross_val_score(model, data_2bit, target, cv=5).mean()

precision_levels = ['Full', 8, 4, 2]
model_accuracies = {
    'Decision Tree': [accuracy_full['Decision Tree'], accuracy_8bit['Decision Tree'], accuracy_4bit['Decision Tree'], accuracy_2bit['Decision Tree']],
    'k-NN': [accuracy_full['k-NN'], accuracy_8bit['k-NN'], accuracy_4bit['k-NN'], accuracy_2bit['k-NN']],
    'SVM': [accuracy_full['SVM'], accuracy_8bit['SVM'], accuracy_4bit['SVM'], accuracy_2bit['SVM']]
}

for model, accuracies in model_accuracies.items():
    plt.plot(precision_levels, accuracies, marker='o', label=model)

plt.xlabel('Precision Level (bits)')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Precision Level')
plt.legend()
plt.show()

