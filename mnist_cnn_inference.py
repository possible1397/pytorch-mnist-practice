import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

# ✅ 與原訓練一致的 CNN 結構
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

# ✅ 載入模型結構與權重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("model_cnn.pt", map_location=device))
model.eval()

# ✅ 載入一張 MNIST 測試圖片（也可以替換為自己的圖片）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = MNIST(root="data", train=False, download=True, transform=transform)
image, label = test_dataset[0]  # 換成別張可改索引

# ✅ 加入 batch 維度後送入模型
with torch.no_grad():
    image_tensor = image.unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]
    output = model(image_tensor)
    pred = torch.argmax(output, dim=1).item()

# ✅ 顯示影像與預測結果
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Prediction: {pred} (Label: {label})")
plt.axis("off")
plt.show()
