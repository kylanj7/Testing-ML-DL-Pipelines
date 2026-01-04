import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb

def main():
    wandb.init(project="ReLU-SGD-BCE-Test-Deep")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Generate synthetic sine wave data
    np.random.seed(42)
    n_samples = 10000
    x1 = np.random.randn(n_samples, 1)
    y1 = np.sin(x1.flatten() * 3) > 0.5
    labels = y1.reshape(-1, 1)

    # Step 2: Create XOR-like binary classification features
    feature1 = torch.tensor(np.random.randn(100, 2), dtype=torch.float32)
    label_bin = torch.tensor([1 if x*y < 0 else 0 for x, y in feature1], dtype=torch.float32).unsqueeze(1)

    # Step 3: Split data into train and test sets (80% test, 20% train)
    X_train, X_test, y_train, y_test = train_test_split(
        feature1, label_bin, test_size=0.8, random_state=4
    )

    class BinaryClassifier(nn.Module):
        def __init__(self, input_dim):
            super(BinaryClassifier, self).__init__()
            self.hidden_layer1 = nn.Linear(input_dim, 2048)
            self.hidden_layer2 = nn.Linear(2048, 1024)
            self.hidden_layer3 = nn.Linear(1024, 1024)
            self.hidden_layer4 = nn.Linear(1024, 512)
            self.hidden_layer5 = nn.Linear(512, 512)
            self.hidden_layer6 = nn.Linear(512, 256)
            self.hidden_layer7 = nn.Linear(256, 256)
            self.hidden_layer8 = nn.Linear(256, 128)
            self.hidden_layer9 = nn.Linear(128, 128)
            self.hidden_layer10 = nn.Linear(128, 64)
            self.output_layer = nn.Linear(64, 1)
        def forward(self, x):
            x = torch.relu(self.hidden_layer1(x))
            x = torch.relu(self.hidden_layer2(x))
            x = torch.relu(self.hidden_layer3(x))
            x = torch.relu(self.hidden_layer4(x))
            x = torch.relu(self.hidden_layer5(x))
            x = torch.relu(self.hidden_layer6(x))
            x = torch.relu(self.hidden_layer7(x))
            x = torch.relu(self.hidden_layer8(x))
            x = torch.relu(self.hidden_layer9(x))
            x = torch.relu(self.hidden_layer10(x))
            x = self.output_layer(x)
            return x

    # Step 4: Move data to GPU device
    X_train_torch = X_train
    y_train_torch = y_train
    X_test_torch = X_test.to(device)
    y_test_torch = y_test.to(device)

    input_dim = X_train.shape[1]
    model = BinaryClassifier(input_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 500
    batch_size = 32

    # Step 5: Create DataLoader for batched training with multi-threading
    train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Log hyperparameters and model architecture
    wandb.config.update({
        "learning_rate": 0.1,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": "SGD",
        "loss_function": "BCEWithLogitsLoss",
        "architecture": "10-layer deep neural network",
        "activation": "ReLU",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "input_dim": input_dim,
        "device": str(device)
    })

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

    wandb.finish()

if __name__ == '__main__':
    main()
