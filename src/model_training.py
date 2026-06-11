import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from logger.log_config import setup_logger
from utils.config import load_params
import numpy as np
import os


class FraudDetectorCNN(nn.Module):
    def __init__(self):
        super(FraudDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def training():
    params = load_params()
    logger = setup_logger()

    try:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("model training has been started")

        X_train = np.load(os.path.join(params["artifacts"]["processed_data"], "X_train.npy"))
        y_train = np.load(os.path.join(params["artifacts"]["processed_data"], "y_train.npy"))

        logger.info(f"training data loaded: X_train shape {X_train.shape}")

        X_train = np.transpose(X_train, (0, 3, 1, 2))
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params["training"]["batch_size"], shuffle=True)

        model = FraudDetectorCNN().to(device) 
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["training"]["learning_rate"])

        logger.info("model initialized and moved to device")

        epochs = params["training"]["epochs"]

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device) 
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - Accuracy: {accuracy:.2f}%")

        logger.info("model training completed successfully")

        model_path = params["artifacts"]["models"]
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_path, "fraud_detector.pt"))
        logger.info(f"model saved to: {model_path}")

    except Exception as e:
        logger.error(f"model training was not successful because {e}")
        raise


if __name__ == "__main__":
    training()
