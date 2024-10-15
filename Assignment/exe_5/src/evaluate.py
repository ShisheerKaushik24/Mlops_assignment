import time
import torch
from src.model import train_model
from src.quantization import quantize_model
from src.data_loader import load_data

def evaluate_model(model, X_test, y_test):
    model.eval()
    start_time = time.time()
    outputs = model(torch.tensor(X_test, dtype=torch.float32))
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == torch.tensor(y_test)).sum().item() / len(y_test)
    inference_time = time.time() - start_time
    model_size = sum(p.numel() for p in model.parameters()) * 4  # float32: 4 bytes
    return accuracy, model_size, inference_time

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('digits')
    input_size = X_train.shape[1]
    num_classes = len(set(y_train))

    model = train_model(X_train, y_train, input_size, num_classes)
    quantized_model = quantize_model(model)

    print("Evaluating original model")
    accuracy, model_size, inference_time = evaluate_model(model, X_test, y_test)
    print(f"Original Model - Accuracy: {accuracy}, Size: {model_size / 1e6:.2f} MB, Inference Time: {inference_time:.6f} seconds")

    print("Evaluating quantized model")
    accuracy_q, model_size_q, inference_time_q = evaluate_model(quantized_model, X_test, y_test)
    print(f"Quantized Model - Accuracy: {accuracy_q}, Size: {model_size_q / 1e6:.2f} MB, Inference Time: {inference_time_q:.6f} seconds")
