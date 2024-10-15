import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.data_loader import load_data

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_model(X_train, y_train, input_size, num_classes, epochs=100):
    model = LogisticRegression(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('digits')
    input_size = X_train.shape[1]
    num_classes = len(set(y_train))
    model = train_model(X_train, y_train, input_size, num_classes)
