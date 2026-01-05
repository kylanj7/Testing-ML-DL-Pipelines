import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

GPU1 = torch.device('cuda:0') # 3090
GPU2 = torch.device('cuda:1') # 3090ti

torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"

def main():
    wandb.init(project="ReLU-SGD-BCE-Test-Deep")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")

    # Step 1: Generate synthetic sine wave data
    np.random.seed(42)
    n_samples = 100000000
    x1 = np.random.randn(n_samples, 1)
    y1 = np.sin(x1.flatten() * 3) > 0.5
    labels = y1.reshape(-1, 1)

    # Step 2: Create XOR-like binary classification features
    feature1 = torch.tensor(np.random.randn(100, 2), dtype=torch.float32)
    label_bin = torch.tensor([1 if x*y < 0 else 0 for x, y in feature1], dtype=torch.float32).unsqueeze(1)

    # Step 3: Split data into train and test sets (80% test, 20% train) using PyTorch
    n_samples = feature1.size(0)
    n_train = int(n_samples * 0.2)
    indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(4))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = feature1[train_indices]
    y_train = label_bin[train_indices]
    X_test = feature1[test_indices]
    y_test = label_bin[test_indices]

    class BinaryClassifier(nn.Module):
        def __init__(self, input_dim):
            super(BinaryClassifier, self).__init__()
            self.hidden_layer1 = nn.Linear(input_dim, 5120)
            self.hidden_layer2 = nn.Linear(5120, 2560)
            self.hidden_layer3 = nn.Linear(2560, 2560)
            self.hidden_layer4 = nn.Linear(2560, 2560)
            self.hidden_layer5 = nn.Linear(2560, 2560)
            self.hidden_layer6 = nn.Linear(2560, 2560)
            self.hidden_layer7 = nn.Linear(2560, 2560)
            self.hidden_layer8 = nn.Linear(2560, 2560)
            self.hidden_layer9 = nn.Linear(2560, 2560)
            self.hidden_layer10 = nn.Linear(2560, 1280)
            self.hidden_layer11 = nn.Linear(1280, 1280)
            self.hidden_layer12 = nn.Linear(1280, 640)
            self.output_layer = nn.Linear(640, 1)
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
            x = torch.relu(self.hidden_layer11(x))
            x = torch.relu(self.hidden_layer12(x))
            x = self.output_layer(x)
            return x

    # Step 4: Keep data on CPU for DataLoader
    X_train_torch = X_train
    y_train_torch = y_train
    X_test_torch = X_test.to(device)
    y_test_torch = y_test.to(device)

    input_dim = X_train.shape[1]
    model = BinaryClassifier(input_dim)

    # Add DataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1])

    model = model.to(device)

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
        "architecture": "12-layer deep neural network",
        "activation": "ReLU",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "input_dim": input_dim,
        "device": str(device),
        "num_gpus": torch.cuda.device_count(),
        "tf32_enabled": torch.backends.cuda.matmul.allow_tf32
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
