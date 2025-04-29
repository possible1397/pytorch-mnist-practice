import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 設定訓練參數
batch_size = 64
learning_rate = 0.01
epochs = 5

# 轉換器：把圖片轉為 Tensor，並正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下載 MNIST 資料集
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 建立模型
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = SimpleNN()

# 損失函數與優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 訓練迴圈
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 測試準確率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x)
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

print(f"測試集準確率: {100 * correct / total:.2f}%")
