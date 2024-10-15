import torch
from torch.quantization import quantize_dynamic
from src.model import train_model
from src.data_loader import load_data

def quantize_model(model):
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('digits')
    input_size = X_train.shape[1]
    num_classes = len(set(y_train))
    model = train_model(X_train, y_train, input_size, num_classes)
    quantized_model = quantize_model(model)
    print("Model quantized successfully")
