# CIFAR-10 dataset으로 이미지 분류 실습 - PyTorch 버전
# CIFAR-10은 총 10개의 레이블로 구성된 6만 장 컬러 이미지 dataset
# train: 5만 장, test: 1만 장
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# GPU 사용 가능하면 cuda, 아니면 cpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 1) 클래스 이름 정의
CLASSES = np.array([
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
])

NUM_CLASSES = 10

# 2) 데이터 준비
# PyTorch 이미지 shape: (batch, channel, height, width) 예: (50000, 3, 32, 32)
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
print("train data 개수:", len(train_dataset))
print("test data 개수:", len(test_dataset))

# 첫 번째 데이터 확인
x0, y0 = train_dataset[0]

print(x0.shape)     # torch.Size([3, 32, 32])
print(y0)           # 정수 라벨
print(CLASSES[y0])  # 라벨 이름

# 3) 이미지 시각화
# PyTorch 이미지 shape은 (C, H, W) matplotlib의 imshow()는 (H, W, C)를 기대하므로
# permute(1, 2, 0)으로 차원 순서를 바꿔야 한다.

plt.figure(figsize=(6, 2))

for i in range(3):
    img, label = train_dataset[i]

    # (3, 32, 32) -> (32, 32, 3)
    img = img.permute(1, 2, 0).numpy()

    plt.subplot(1, 3, i + 1)
    plt.imshow(img, interpolation="bicubic")
    plt.title(CLASSES[label])
    plt.axis("off")

plt.show()

# 4) DataLoader 생성
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5) 모델 정의
class CifarDenseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=3 * 32 * 32, out_features=256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),

            # softmax 사용하지 않음
            nn.Linear(in_features=128, out_features=NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

model = CifarDenseModel().to(device)
print(model)

# 6) 손실 함수와 최적화 함수 정의
# PyTorch: nn.CrossEntropyLoss()
#   y는 one-hot encoding 하지 않고 정수 라벨 그대로 사용
# 예: cat -> 3,  truck -> 9
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# 7) 학습 함수 정의
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

        # 예측 클래스 계산
        preds = torch.argmax(outputs, dim=1)

        # 정답 개수 누적
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count

    return avg_loss, avg_acc

# 8) 평가 함수 정의
def evaluate(model, loader, criterion, device):
    model.eval()   # 평가 모드

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # 평가 시에는 gradient 계산 불필요
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_count += y_batch.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count

    return avg_loss, avg_acc

# 9) 모델 학습
epochs = 20
history = { "loss": [], "accuracy": [] }

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"loss: {train_loss:.4f}, acc: {train_acc:.4f}"
    )

# 10) 모델 평가
test_loss, test_acc = evaluate(model, test_loader, criterion, device )
print("test acc : %.4f" % test_acc)
print("test loss : %.4f" % test_loss)

# 11) 예측 : 여기서는 test_dataset의 앞 10개 이미지를 예측한다.
model.eval()

test_images = []
test_labels = []

for i in range(10):
    img, label = test_dataset[i]
    test_images.append(img)
    test_labels.append(label)

# list에 담긴 Tensor들을 하나의 batch Tensor로 변환
# 각각 shape: (3, 32, 32), 변환 후 shape: (10, 3, 32, 32)
x_batch = torch.stack(test_images).to(device)

with torch.no_grad():
    outputs = model(x_batch)
    # 각 이미지별로 가장 큰 출력값을 가진 클래스 index
    pred_idx = torch.argmax(outputs, dim=-1)

# Tensor를 CPU로 가져온 뒤 numpy 배열로 변환
pred_idx = pred_idx.cpu().numpy()
actual_idx = np.array(test_labels)

pred = CLASSES[pred_idx]
actual = CLASSES[actual_idx]
print("예측값 : ", pred)
print("실제값 : ", actual)
print("분류 실패 수 : ", (pred != actual).sum())

# 12) 예측 결과 시각화
fig = plt.figure(figsize=(15, 3))
# subplot 사이 간격 조정, hspace: 위아래 간격, wspace: 좌우 간격
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(range(10)):
    img = test_images[idx]

    # PyTorch image: (C, H, W),  matplotlib image: (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    ax = fig.add_subplot(1, 10, i + 1)
    ax.axis("off")
    ax.imshow(img)

    # transAxes:
    # 이미지 픽셀 좌표가 아니라 subplot 영역 기준 좌표 사용
    # x 좌표: 0.0 -> 왼쪽, 0.5 -> 가운데, 1.0 -> 오른쪽
    # y 좌표: 0.0 -> 아래쪽, 1.0 -> 위쪽, 음수 -> subplot 아래쪽 바깥 영역
    ax.text(
        0.5, -0.35, "pred=" + str(pred[idx]),
        fontsize=10, ha="center", transform=ax.transAxes
    )
    ax.text(
        0.5, -0.7, "act=" + str(actual[idx]),
        fontsize=10, ha="center", transform=ax.transAxes
    )

plt.show()

# 13) 학습 곡선 시각화
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="train acc")
plt.title("Train Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="train loss")
plt.title("Train Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 현재 모델은 컬러 이미지 분류에 대한 정확도가 낮을 수 있음
# 이유: Flatten + Linear만 사용했기 때문
#   이미지의 공간적 특징, 즉 위치 관계와 패턴을 충분히 학습하기 어렵다.
#   CIFAR-10 같은 이미지 데이터에는 일반적으로 CNN 구조가 더 적합하다.