import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
wandb.init(project="ReLU-SGD-BCE-Test")

np.random.seed(42)
n_samples = 1000000000
x1 = np.random.randn(n_samples, 1)
y1 = np.sin(x1.flatten() * 3) > 0.5
labels = y1.reshape(-1, 1)

feature1 = torch.tensor(np.random.randn(100, 2), dtype=torch.float32)
label_bin = torch.tensor([1 if x*y < 0 else 0 for x, y in feature1], dtype=torch.float32).unsqueeze(1)
X_train, X_test, y_train, y_test = train_test_split(
    feature1, label_bin, test_size=0.8, random_state=4
)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.hidden_layer1 = nn.Linear(input_dim, 512)
        self.hidden_layer2 = nn.Linear(512, 254)
        self.output_layer = nn.Linear(254, 1)
    def forward(self, x):
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

X_train_torch = X_train
y_train_torch = y_train
X_test_torch = X_test
y_test_torch = y_test

input_dim = X_train.shape[1]
model = BinaryClassifier(input_dim)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 500
batch_size = 32

train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
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
