# 사전 학습 가중치를 사용하지 않고, MobileNetV2 구조만 가져와서 CIFAR-10의 10개 클래스로 처음부터 학습
# MobileNetV2 구조를 사용하여 CIFAR-10 분류하기 - PyTorch 버전
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 재현성을 위한 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. CIFAR-10 데이터셋 불러오기
# PyTorch에서는 transforms.ToTensor()가 다음 작업을 자동 수행함
# 1) 이미지를 Tensor로 변환
# 2) 픽셀값을 0~255에서 0~1 범위로 정규화
# 3) 이미지 차원을 (H, W, C)에서 (C, H, W)로 변경
# CIFAR-10 원본 이미지 크기: PyTorch:(3, 32, 32)
transform = transforms.ToTensor()

train_full_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
print("전체 학습 데이터 수:", len(train_full_dataset))
print("테스트 데이터 수:", len(test_dataset))

# 2. 학습 데이터 / 검증 데이터 분리
train_size = int(len(train_full_dataset) * 0.8)
val_size = len(train_full_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print("학습 데이터 수:", len(train_dataset))
print("검증 데이터 수:", len(val_dataset))

# 3. DataLoader 생성
batch_size = 64

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# 4. CIFAR-10 클래스 이름
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 5. 데이터 확인 시각화
images, labels = next(iter(train_loader))

plt.figure(figsize=(6, 6))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    # PyTorch 이미지 shape: (C, H, W) : matplotlib은 (H, W, C)를 사용하므로 permute로 차원 변경
    img = images[i].permute(1, 2, 0)
    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()
print("라벨:", labels[:16])

# 6. MobileNetV2 모델 생성
model = mobilenet_v2(
    weights=None, num_classes=10
)
model = model.to(device)
print(model)

# 7. 파라미터 수 확인
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터 수: {total:,}")
    print(f"학습 가능 파라미터 수: {trainable:,}")

count_params(model)

# 8. 손실 함수와 옵티마이저 정의
# TensorFlow 코드에서는 y_train을 원핫인코딩한 뒤 categorical_crossentropy를 사용했음
# PyTorch에서는 보통 원핫인코딩을 하지 않고, 정수 라벨을 그대로 사용함
# 예: airplane -> 0, automobile -> 1 ... truck -> 9
criterion = nn.CrossEntropyLoss()
# TensorFlow의 optimizer='adam'과 같은 역할
optimizer = optim.Adam(model.parameters())

# 9. 학습 함수 정의
def train_model(model, train_loader, val_loader, epochs=10):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        # 학습 단계
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            # 데이터를 CPU 또는 GPU로 이동
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 이전 mini-batch에서 계산된 gradient 초기화
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs, dim=1)

            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 검증 단계
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

# 10. 테스트 평가 함수 정의
def evaluate_model(model, test_loader):
    model.eval()
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

# 11. 모델 학습
history = train_model(
    model=model, train_loader=train_loader, val_loader=val_loader, epochs=10
)

# 12. 테스트 데이터 평가
loss, acc = evaluate_model(model, test_loader)
print(f"test loss: {loss:.4f}, test acc: {acc:.4f}")

# 13. 성능 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["loss"], "b-", label="loss")
plt.plot(history["val_loss"], "r--", label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], "b-", label="accuracy")
plt.plot(history["val_accuracy"], "r--", label="val_accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()

plt.tight_layout()
plt.show()