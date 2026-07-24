# Fashion MNIST 모델 성능 비교
# 1) Conv + Dense
# 2) Conv + Pooling + Dense
# 3) 유명 모델(VGGNet 일부 구조 응용)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# GPU 사용 가능하면 GPU 사용, 아니면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 재현성을 위한 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Fashion MNIST 데이터셋 불러오기
transform = transforms.ToTensor()

train_full_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
print("전체 학습 데이터 수:", len(train_full_dataset))
print("테스트 데이터 수:", len(test_dataset))

# 2. 학습 데이터 / 검증 데이터 분리
# TensorFlow 코드의 validation_split=0.25와 동일하게 구성. 전체 60,000개 중 75%는 학습, 25%는 검증
train_size = int(len(train_full_dataset) * 0.75)
val_size = len(train_full_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print("학습 데이터 수:", len(train_dataset))
print("검증 데이터 수:", len(val_dataset))

# 3. DataLoader 생성
batch_size = 128

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# 4. 데이터 확인 시각화
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

images, labels = next(iter(train_loader))

plt.figure(figsize=(5, 5))
for i in range(16):
    plt.subplot(4, 4, i + 1)

    # PyTorch 이미지 shape: (채널, 높이, 너비), matplotlib에서 보기 위해 (28, 28)로 변환
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()
print("라벨:", labels[:16].numpy())


# 5. 모델 1: Conv + Dense
class ConvDenseModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Keras 원본 코드에서는 Conv2D에 activation을 지정하지 않았음
        # 따라서 여기서도 Conv2D 뒤에 ReLU를 넣지 않고 동일하게 구성
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        )

        # 입력 이미지 크기: 28 x 28   Conv 3x3 padding 없음
        # 28 -> 26 -> 24 -> 22   최종 feature map: 64 x 22 x 22
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # PyTorch의 CrossEntropyLoss는 내부적으로 softmax를 포함함
            # 따라서 마지막 층에는 softmax를 넣지 않음
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# 6. 모델 2: Conv + Pooling + Dense
class ConvPoolingDenseModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Keras 원본 코드와 동일하게 Conv 뒤 activation 없음  Conv → MaxPool 구조
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 크기 변화 : 입력: 1 x 28 x 28
        # Conv: 16 x 26 x 26
        # Pool: 16 x 13 x 13
        # Conv: 32 x 11 x 11
        # Pool: 32 x 5 x 5
        # Conv: 64 x 3 x 3
        # Pool: 64 x 1 x 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# 7. 모델 3: VGGNet 일부 구조 응용 모델
class VGGStyleModel(nn.Module):
    def __init__(self):
        super().__init__()

        # VGGNet의 특징:
        # - 작은 3x3 Conv 필터를 여러 번 사용
        # - Conv를 여러 번 쌓은 뒤 MaxPooling으로 크기 축소
        # - 뒤쪽에 Dense 층을 깊게 구성
        self.conv_layers = nn.Sequential(
            # Keras padding='same'과 유사하게 padding=1 사용
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Keras padding='valid'와 동일하게 padding 없음
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )

        # 크기 변화 : 입력: 1 x 28 x 28
        # Conv same: 32 x 28 x 28
        # Conv same: 64 x 28 x 28
        # Pool: 64 x 14 x 14
        # Conv same: 128 x 14 x 14
        # Conv valid: 256 x 12 x 12
        # Pool: 256 x 6 x 6
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# 8. 모델 파라미터 수 확인 함수
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"전체 파라미터 수: {total:,}")
    print(f"학습 가능 파라미터 수: {trainable:,}")

# 9. 학습 함수 정의
def train_model(model, train_loader, val_loader, epochs=15):
    model = model.to(device)

    # 다중 클래스 분류 손실 함수
    # CrossEntropyLoss는 내부적으로 Softmax + Negative Log Likelihood를 처리함
    criterion = nn.CrossEntropyLoss()

    # Keras의 optimizer='adam'과 동일한 역할
    optimizer = optim.Adam(model.parameters())

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        # 학습 모드
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            # 데이터를 CPU/GPU 장치로 이동
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            # 예측 클래스 계산
            _, predicted = torch.max(outputs, dim=1)

            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 검증 모드
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 검증 시에는 gradient 계산이 필요 없음
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)

                _, predicted = torch.max(outputs, dim=1)

                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    return history

# 10. 평가 함수 정의
def evaluate_model(model, test_loader):
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            test_loss += loss.item() * x_batch.size(0)

            _, predicted = torch.max(outputs, dim=1)

            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()

    test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    return test_loss, test_acc

# 11. 성능 시각화 함수
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], "b-", label="loss")
    plt.plot(history["val_loss"], "r--", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title + " - Loss")
    plt.legend()

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], "b-", label="accuracy")
    plt.plot(history["val_accuracy"], "r--", label="val_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title + " - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


print("1) Conv + Dense 모델")
model1 = ConvDenseModel()
count_params(model1)

history1 = train_model(
    model1, train_loader, val_loader, epochs=15
)

loss1, acc1 = evaluate_model(model1, test_loader)
print(f"[Conv + Dense] test loss: {loss1:.4f}, test acc: {acc1:.4f}")

plot_history(history1, "Conv + Dense")


print("2) Conv + Pooling + Dense 모델")
model2 = ConvPoolingDenseModel()
count_params(model2)

history2 = train_model(
    model2, train_loader, val_loader, epochs=15
)

loss2, acc2 = evaluate_model(model2, test_loader)
print(f"[Conv + Pooling + Dense] test loss: {loss2:.4f}, test acc: {acc2:.4f}")

plot_history(history2, "Conv + Pooling + Dense")

# 모델 3 학습 및 평가
print("3) VGGStyle 모델")
model3 = VGGStyleModel()
count_params(model3)

history3 = train_model(
    model3, train_loader, val_loader, epochs=15
)

loss3, acc3 = evaluate_model(model3, test_loader)
print(f"[VGGStyle] test loss: {loss3:.4f}, test acc: {acc3:.4f}")

plot_history(history3, "VGGStyle")

print("최종 성능 비교")
results = [
    ["Conv + Dense", loss1, acc1],
    ["Conv + Pooling + Dense", loss2, acc2],
    ["VGGStyle", loss3, acc3]
]

for name, loss, acc in results:
    print(f"{name:25s} | loss: {loss:.4f} | acc: {acc:.4f}")