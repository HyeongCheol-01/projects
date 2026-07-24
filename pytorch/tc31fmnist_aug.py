# FashionMNIST dataset에 이미지 보강 후 분류
# - torchvision.datasets.FashionMNIST 사용
# - torchvision.transforms로 이미지 보강
# - DataLoader로 batch 데이터 공급
# - nn.Module로 CNN 모델 정의
# - 학습 루프 직접 작성
# - EarlyStopping, Checkpoint 직접 구현

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# FashionMNIST 클래스 이름
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 5. Transform 정의
train_transform = transforms.Compose([
    # FashionMNIST는 PIL Image 형태로 로딩된다. 회전: -10도 ~ +10도
    transforms.RandomRotation(degrees=10),

    # 평행 이동, 확대/축소, shear 변환
    # translate=(0.1, 0.1): 가로/세로 최대 10% 이동
    # scale=(0.9, 1.1): 90% 축소 ~ 110% 확대, shear=0.5: 기울이기
    transforms.RandomAffine(
        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.5
    ),
    transforms.RandomHorizontalFlip(p=0.5),  # 수평 반전
    # 수직 반전
    # FashionMNIST에서는 실제 의류 이미지가 뒤집히면 어색
    transforms.RandomVerticalFlip(p=0.5),

    # PIL Image -> Tensor 변환
    # shape: (H, W, C) -> (C, H, W),  pixel: 0~255 -> 0~1
    transforms.ToTensor()
])

test_transform = transforms.Compose([transforms.ToTensor()])

# 6. FashionMNIST 데이터셋 로딩
train_full_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=train_transform
)

test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=test_transform
)

# 7. Train / Validation 분리
train_size = int(len(train_full_dataset) * 0.8)
val_size = len(train_full_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
)
print("train 데이터 수:", len(train_dataset))
print("val 데이터 수:", len(val_dataset))
print("test 데이터 수:", len(test_dataset))

# 8. DataLoader 생성
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

# 9. 원본 이미지 100개 시각화
# test_transform을 사용하는 별도 dataset을 만들어 증강되지 않은 원본 FashionMNIST 이미지를 확인한다.
original_train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=test_transform
)

plt.figure(figsize=(10, 10))
for c in range(100):
    img, label = original_train_dataset[c]
    # img shape: (1, 28, 28), matplotlib 출력을 위해 (28, 28)로 변환
    plt.subplot(10, 10, c + 1)
    plt.axis("off")
    plt.imshow(img.squeeze(0), cmap="gray")
plt.show()

# 10. 증강 이미지 일부 시각화
# 같은 원본 이미지라도 train_transform을 통과할 때마다 랜덤 증강이 적용되어 다른 이미지처럼 보일 수 있다.
plt.figure(figsize=(12, 3))

for i in range(10):
    img, label = train_full_dataset[i]
    plt.subplot(1, 10, i + 1)
    plt.imshow(img.squeeze(0), cmap="gray")
    plt.axis("off")
    plt.title(str(label))
plt.show()

# 11. CNN 모델 정의 : Conv2d -> MaxPool2d -> Dropout : Linear 사용
# 주의: PyTorch의 CrossEntropyLoss는 내부적으로 softmax를 포함한다. 따라서 마지막 layer에 softmax를 넣지 않는다.
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.features = nn.Sequential(
            # 입력 shape: (batch, 1, 28, 28)
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3,padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 출력 shape: (batch, 32, 28, 28)

            nn.Dropout(p=0.1),  # 출력 shape: (batch, 32, 14, 14)

            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 출력 shape: (batch, 64, 14, 14)
            nn.Dropout(p=0.1)  # 출력 shape: (batch, 64, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 64),  # 64 * 7 * 7 = 3136
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(32, 10)   # 10개 클래스 분류
        )

    def forward(self, x):
        x = self.features(x) 
        x = self.classifier(x)
        return x

model = FashionCNN().to(device)
print(model)

# 12. 손실 함수와 Optimizer 설정 
# CrossEntropyLoss는 다음을 내부적으로 처리한다. softmax + negative log likelihood
# 따라서 y는 one-hot encoding하지 않고 정수 라벨 그대로 사용한다.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 13. 학습 함수 정의
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    1 epoch 동안 모델 학습 흐름:
        1. model.train()
        2. optimizer.zero_grad()
        3. outputs = model(images)
        4. loss = criterion(outputs, labels)
        5. loss.backward()
        6. optimizer.step()
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        # 예측 클래스
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc

# 14. 평가 함수 정의
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc

# 15. 모델 저장 디렉토리 생성
MODEL_DIR = "./fmnist/"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# 16. EarlyStopping + ModelCheckpoint 직접 구현
epochs = 100
patience = 5

best_val_loss = float("inf")
patience_counter = 0
best_model_path = os.path.join(MODEL_DIR, "best_fashion_cnn.pth")

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

# 17. 모델 학습 실행
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device
    )

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(
        f"Epoch [{epoch + 1:03d}/{epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # val_loss가 좋아지면 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save( model.state_dict(), best_model_path )
        print(f"  -> best model 저장: {best_model_path}")
    else:
        patience_counter += 1
        print(f"  -> val_loss 개선 없음: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("EarlyStopping 실행")
            break

# 18. best model 불러오기
model.load_state_dict(
    torch.load(best_model_path, map_location=device)
)
model.to(device)

# 19. Test 정확도 확인
test_loss, test_acc = evaluate(
    model, test_loader, criterion, device
)
print(f"test acc : {test_acc:.4f}")

# 20. 학습 과정 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, marker="o", label="train acc")
plt.plot(val_acc_history, marker="s", label="val acc")
plt.xlabel("epochs")
plt.ylim(0.5, 1)
plt.legend(loc="lower right")
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, marker="o", label="train loss")
plt.plot(val_loss_history, marker="s", label="val loss")
plt.xlabel("epochs")
plt.legend(loc="upper right")
plt.title("Loss")
plt.show()

# 21. 테스트 이미지 예측 시각화
model.eval()

plt.figure(figsize=(16, 4))
with torch.no_grad():
    for i in range(10):
        image, label = test_dataset[i]
        input_image = image.unsqueeze(0).to(device)
        output = model(input_image)
        pred = output.argmax(dim=1).item()

        plt.subplot(1, 10, i + 1)
        plt.imshow(image.squeeze(0), cmap="gray")

        title_color = "black" if pred == label else "red"
        plt.title(
            f"pred:{class_names[pred]}\ntrue:{class_names[label]}",
            fontsize=8,
            color=title_color
        )
        plt.axis("off")

plt.tight_layout()
plt.show()