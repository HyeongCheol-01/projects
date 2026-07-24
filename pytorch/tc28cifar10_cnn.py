# CIFAR10 dataset으로 분류기 작성 + CNN 레이어 추가 - PyTorch 전체 코드

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

torch.manual_seed(0)
np.random.seed(0)

NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

# 2. CIFAR10 데이터셋 로딩
# PyTorch의 CIFAR10 이미지는 PIL Image 형태로 로딩됨
# transforms.ToTensor()를 사용하면 다음 작업이 자동으로 처리됨
# 1) 이미지를 Tensor로 변환
# 2) 픽셀값을 0~255에서 0~1 범위로 정규화
# 3) 이미지 shape을 (H, W, C)에서 (C, H, W)로 변경
# TensorFlow/Keras: (32, 32, 3),  PyTorch:         (3, 32, 32)

transform = transforms.Compose([
    transforms.ToTensor()
])

train_full_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("전체 train 데이터 수:", len(train_full_dataset))
print("test 데이터 수:", len(test_dataset))

# CIFAR10 클래스 이름
CLASSES = np.array([
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
])

# 3. train / validation 데이터 분리
# 전체 train 50000장 중 90%는 학습용, 10%는 검증용으로 사용
train_size = int(len(train_full_dataset) * 0.9)
val_size = len(train_full_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_full_dataset,
    [train_size, val_size]
)
print("train 데이터 수:", len(train_dataset))
print("validation 데이터 수:", len(val_dataset))

# 4. DataLoader 생성 : DataLoader는 데이터를 batch 단위로 꺼내주는 역할
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 5. CNN Block 정의 : nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# 6. CNN 모델 정의
class Cifar10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Stage 1
        # 입력 shape: (batch_size, 3, 32, 32)
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2)
        )
        # 출력 shape: (batch_size, 32, 16, 16)

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2)
        )
        # 출력 shape: (batch_size, 64, 8, 8)

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2)
        )
        # 출력 shape: (batch_size, 128, 4, 4)

        # Keras의 GlobalAveragePooling2D와 같은 역할
        #
        # (batch_size, 128, 4, 4)
        # -> (batch_size, 128, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),

            # AdaptiveAvgPool2d 이후 flatten하면 feature 수는 128개
            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Dropout(p=0.2),

            # PyTorch의 CrossEntropyLoss는 내부적으로 Softmax 계산을 포함함
            # 따라서 마지막에 softmax를 사용하지 않음
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.global_avg_pool(x)

        # (batch_size, 128, 1, 1) -> (batch_size, 128)
        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)
        return x


model = Cifar10CNN(num_classes=NUM_CLASSES).to(device)
print(model)

# 7. Loss 함수와 Optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 8. 정확도 계산 함수
def calculate_accuracy(outputs, labels):
    # outputs shape: (batch_size, 10)
    # labels shape:  (batch_size)

    # 각 sample마다 가장 큰 출력값을 가진 class index 선택
    _, preds = torch.max(outputs, dim=1)

    # 예측값과 실제값이 같은 개수 계산
    correct = (preds == labels).sum().item()

    return correct

# 9. EarlyStopping 클래스 구현
class EarlyStopping:
    def __init__(self, patience=8):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        score = val_acc

        # 첫 번째 epoch에서는 현재 성능을 best로 저장
        if self.best_score is None:
            self.best_score = score

            # 현재 모델 가중치를 복사해서 저장
            self.best_model_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

        # 검증 정확도가 개선되지 않은 경우
        elif score <= self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        # 검증 정확도가 개선된 경우
        else:
            self.best_score = score
            self.best_model_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
            self.counter = 0

    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


early_stopping = EarlyStopping(patience=8)

# 10. 학습 함수
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    # 학습 모드
    # Dropout 활성화, BatchNorm은 batch 통계 사용
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in train_loader:
        # images shape: (batch_size, 3, 32, 32)
        # labels shape: (batch_size)

        images = images.to(device)
        labels = labels.to(device)

        # 이전 batch에서 계산된 gradient 초기화
        optimizer.zero_grad()

        # forward
        outputs = model(images)

        # loss 계산
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        # batch loss 누적
        running_loss += loss.item() * images.size(0)

        # batch 정확도 누적
        running_correct += calculate_accuracy(outputs, labels)

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc

# 11. 검증 / 테스트 함수
def evaluate(model, data_loader, criterion, device):
    # 평가 모드
    # Dropout 비활성화, BatchNorm은 저장된 이동 평균값 사용
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    # 평가 시에는 gradient 계산이 필요 없음
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_correct += calculate_accuracy(outputs, labels)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc

# 12. 전체 학습 진행
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
        f"train_loss: {train_loss:.4f}, "
        f"train_acc: {train_acc:.4f}, "
        f"val_loss: {val_loss:.4f}, "
        f"val_acc: {val_acc:.4f}"
    )

    # EarlyStopping 검사
    early_stopping(val_acc, model)

    if early_stopping.early_stop:
        print("EarlyStopping 발생 - 학습 중단")
        break


# 가장 좋은 검증 정확도를 기록한 모델 가중치 복원
early_stopping.restore_best_weights(model)

# 13. Test 데이터 평가
test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion,
    device
)

print("test loss : %.4f" % test_loss)
print("test acc  : %.4f" % test_acc)

# 14. Test 데이터 중 앞 10개 예측
model.eval()

# test_loader에서 첫 번째 batch만 가져오기
images, labels = next(iter(test_loader))

# 앞 10개만 사용
images_10 = images[:10].to(device)
labels_10 = labels[:10]

with torch.no_grad():
    outputs = model(images_10)

    # outputs는 softmax가 적용되지 않은 raw score(logit)
    # 가장 큰 값을 가진 index가 예측 class
    preds_10 = torch.argmax(outputs, dim=1).cpu()

pred = CLASSES[preds_10.numpy()]
actual = CLASSES[labels_10.numpy()]

print("예측값 : ", pred)
print("실제값 : ", actual)
print("분류 실패 수 : ", (pred != actual).sum())

# 15. 예측 결과 시각화
fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(10):
    # PyTorch 이미지 shape: (C, H, W)
    # matplotlib 출력 shape: (H, W, C)
    img = images[i].permute(1, 2, 0).numpy()

    ax = fig.add_subplot(1, 10, i + 1)
    ax.axis("off")
    ax.imshow(img)

    # transAxes:
    # 이미지 픽셀 좌표가 아니라 subplot 영역 기준 좌표 사용
    # x=0.5는 가운데 정렬 위치
    # y=-0.35, -0.7은 이미지 아래쪽 위치
    ax.text(
        0.5,
        -0.35,
        "pred=" + str(pred[i]),
        fontsize=10,
        ha="center",
        transform=ax.transAxes
    )

    ax.text(
        0.5,
        -0.7,
        "act=" + str(actual[i]),
        fontsize=10,
        ha="center",
        transform=ax.transAxes
    )

plt.show()

# 16. 학습 과정 시각화
plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()