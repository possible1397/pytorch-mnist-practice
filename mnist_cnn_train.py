import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ✅ 自動選擇 GPU 或 CPU 執行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 超參數設定
batch_size = 64
learning_rate = 0.01
epochs = 5

# ✅ 影像前處理（正規化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ✅ 載入 MNIST 資料集
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ✅ CNN 模型定義
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 輸入: 1x28x28 → 輸出: 32x26x26
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → 32x13x13
            nn.Conv2d(32, 64, kernel_size=3),  # → 64x11x11
            nn.ReLU(),
            nn.MaxPool2d(2)                    # → 64x5x5
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),                      # 攤平成 64*5*5 = 1600
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

# ✅ 建立模型 + 損失函數 + Optimizer
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ✅ 訓練過程
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# ======== 儲存模型權重 (.pt) ========
torch.save(model.state_dict(), 'model_cnn.pt')
print("✅ 模型已成功儲存為 model_cnn.pt")

# ✅ 測試模型準確率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

print(f"測試集準確率: {100 * correct / total:.2f}%")
