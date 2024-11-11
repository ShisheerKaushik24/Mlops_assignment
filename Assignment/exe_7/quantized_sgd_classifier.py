# quantized_sgd_classifier.py
"""
This script demonstrates the use of quantized gradients in a custom SGD classifier.
The dataset used is Breast Cancer, and we compare full-precision accuracy with quantized (2-bit) accuracy.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def quantized_sgd_update(gradient, bits):
    levels = 2 ** bits
    step = 1 / (levels - 1)
    stochastic_gradient = np.floor(gradient / step + np.random.rand(*gradient.shape)) * step
    return stochastic_gradient

class QuantizedSGDClassifier(SGDClassifier):
    def __init__(self, bits=2, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def _fit_one_epoch(self, X, y, coef, intercept, alpha, C, sample_weight):
        for i in range(X.shape[0]):
            gradient = self._compute_gradient(X[i:i+1], y[i:i+1], coef, intercept, alpha, C)
            
            quantized_gradient = quantized_sgd_update(gradient, self.bits)
            
            coef -= quantized_gradient
            intercept -= quantized_gradient
            
            super()._partial_fit(X[i:i+1], y[i:i+1], alpha=alpha, C=C,
                                 loss=self.loss, learning_rate=self.learning_rate, 
                                 max_iter=1, sample_weight=sample_weight)

full_precision_model = SGDClassifier(loss='log_loss', max_iter=100000, tol=1e-3, random_state=42)
full_precision_model.fit(X_train, y_train)
full_precision_predictions = full_precision_model.predict(X_test)
full_precision_accuracy = accuracy_score(y_test, full_precision_predictions)

print(f"Full-precision Logistic Regression Accuracy: {full_precision_accuracy:.4f}")

for bits in [2, 4, 8]:
    quantized_model = QuantizedSGDClassifier(loss='log_loss', max_iter=100000, tol=1e-3, random_state=42, bits=bits)
    quantized_model.fit(X_train, y_train)
    quantized_predictions = quantized_model.predict(X_test)
    quantized_accuracy = accuracy_score(y_test, quantized_predictions)
    
    print(f"Quantized Gradient Logistic Regression Accuracy ({bits}-bit): {quantized_accuracy:.4f}")
