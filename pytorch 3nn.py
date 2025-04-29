import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 建立資料：x = -2 ~ 2, y = x^2
x = torch.unsqueeze(torch.linspace(-2, 2, 100), dim=1)  # shape [100, 1]
y = x.pow(2)

# 三層 MLP 模型
model = nn.Sequential(
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 訓練
for epoch in range(3000):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 預測
model.eval()
with torch.no_grad():
    x_np = x.numpy()
    y_np = y.numpy()
    pred = model(x).numpy()

    # 中間特徵
    mid1 = model[0](x)
    act1 = model[1](mid1)
    mid2 = model[2](act1)
    act2 = model[3](mid2)

# 畫結果
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 預測 vs 真實曲線
axs[0, 0].plot(x_np, y_np, label="Ground Truth: $y = x^2$", color='black')
axs[0, 0].plot(x_np, pred, label="Predicted", linestyle="--")
axs[0, 0].set_title("Model Prediction")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 第一層 Linear 輸出
axs[0, 1].plot(x_np, mid1.numpy())
axs[0, 1].set_title("1st Linear Output (before activation)")
axs[0, 1].grid(True)

# 第一層 ReLU 輸出
axs[1, 0].plot(x_np, act1.numpy())
axs[1, 0].set_title("1st ReLU Output")
axs[1, 0].grid(True)

# 第二層 ReLU 輸出
axs[1, 1].plot(x_np, act2.numpy())
axs[1, 1].set_title("2nd ReLU Output")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
