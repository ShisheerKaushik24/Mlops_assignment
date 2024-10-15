import torch
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data(dataset='MNIST'):
    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        X_train = mnist_train.data
        y_train = mnist_train.targets
        X_test = mnist_test.data
        y_test = mnist_test.targets
        X_train = X_train.view(X_train.size(0), -1).float()
        X_test = X_test.view(X_test.size(0), -1).float()   
        return X_train, X_test, y_train, y_test 
    elif dataset == 'digits':
        digits = load_digits()
        X = digits.data
        y = digits.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":       
    train_data, test_data = load_data('digits')