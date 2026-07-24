# MNIST dataset으로 CNN 실습 - PyTorch 버전

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 0) 장치 설정
# GPU 사용 가능하면 cuda, 아니면 cpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 1) 데이터 준비
transform = transforms.ToTensor()

train_full_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

print(train_full_dataset.data[0])       # 첫 번째 이미지 데이터
print(train_full_dataset.data.shape)    # torch.Size([60000, 28, 28])

# PyTorch CNN 입력 shape:
# TensorFlow : (batch, height, width, channel)
# PyTorch    : (batch, channel, height, width)
x_sample, y_sample = train_full_dataset[0]
print(x_sample.shape)   # torch.Size([1, 28, 28])
print(y_sample)         # label

# 2) train / validation 분리
# 학습 데이터 60000개 중 10%를 검증 데이터로 분리
train_size = int(len(train_full_dataset) * 0.9)
val_size = len(train_full_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(123)
)
print("train size:", len(train_dataset))
print("val size:", len(val_dataset))
print("test size:", len(test_dataset))

# 3) DataLoader 생성
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

# 전체 train 평가용 DataLoader
train_eval_loader = DataLoader(
    train_full_dataset, batch_size=batch_size, shuffle=False
)

# 4) CNN 모델 정의
# 입력 shape: PyTorch: (batch, 1, 28, 28)
# Conv2d padding=1은 TensorFlow의 padding='same'과 유사
# 크기 변화:
#   입력:        1 x 28 x 28
#   Conv2D:     16 x 28 x 28
#   MaxPool2D:  16 x 14 x 14
#   Conv2D:     32 x 14 x 14
#   MaxPool2D:  32 x 7 x 7
#   Flatten:    32 * 7 * 7 = 1568

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.2)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=32 * 7 * 7, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # softmax를 넣지 않음
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

model = CNNModel().to(device)
print(model)

# 5) 손실 함수와 최적화 함수 정의
# 정답 y는 one-hot encoding이 아니라 0~9 정수 label 그대로 사용한다.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 6) 학습 / 평가 함수 정의
def train_one_epoch(model, loader, criterion, optimizer, device):
    # 학습 모드 : Dropout, BatchNorm 등이 학습 방식으로 동작
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

        preds = torch.argmax(outputs, dim=1)
        # 정확도 계산용 누적
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count

    return avg_loss, avg_acc

def evaluate(model, loader, criterion, device):
    # 평가 모드 : Dropout 비활성화, BatchNorm은 평가 방식으로 동작
    model.eval()

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

# 7) EarlyStopping 직접 구현
# Keras의 EarlyStopping과 비슷하게
# val_loss가 patience 횟수만큼 개선되지 않으면 학습 중단
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.best_model_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0

            # 가장 좋은 모델 가중치 저장
            self.best_model_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
            return False
        else:
            self.counter += 1

            if self.counter >= self.patience:
                return True

            return False

    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

# 8) 모델 학습
epochs = 100
early_stopping = EarlyStopping(patience=3)

history = {
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": []
}

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device
    )

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
        f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
    )

    # EarlyStopping 확인
    stop = early_stopping.step(val_loss, model)

    if stop:
        print("EarlyStopping 발생")
        break

# 가장 성능이 좋았던 가중치로 복원
early_stopping.restore_best_weights(model)

# 9) 모델 평가
# 아래 둘의 평가 점수 차이가 크면 과적합을 의심할 수 있다.
train_loss, train_acc = evaluate(
    model, train_eval_loader, criterion, device
)
print(f"train loss {train_loss:.4f}, train acc {train_acc:.4f}")

test_loss, test_acc = evaluate(
    model, test_loader, criterion, device
)
print(f"test loss {test_loss:.4f}, test acc {test_acc:.4f}")

# 10) 모델 저장 및 재로딩
# PyTorch에서는 보통 모델 전체가 아니라 state_dict, 즉 가중치만 저장하는 방식을 권장한다.
SAVE_PATH = "cnn1model.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"모델 저장 {SAVE_PATH}")

# ----------------------------------------
loaded_model = CNNModel().to(device)
loaded_model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
loaded_model.eval()

test_loss, test_acc = evaluate(
    loaded_model, test_loader, criterion, device
)
print(f"[Reloaded] test loss {test_loss:.4f}, test acc {test_acc:.4f}")

# 11) 분류 예측
# 편의상 기존 test_dataset의 첫 번째 자료 사용
idx = 0

x_one, y_true = test_dataset[idx]
print(y_true)  # 보통 7

# 모델 입력은 batch 차원이 필요하므로 unsqueeze(0) 사용
# 기존 shape: (1, 28, 28), 변경 shape: (1, 1, 28, 28)
x_one_batch = x_one.unsqueeze(0).to(device)
loaded_model.eval()

with torch.no_grad():
    logits = loaded_model(x_one_batch)
    # CrossEntropyLoss 학습에서는 softmax 없이 logits를 사용하지만,
    # 확률값을 보고 싶을 때는 softmax를 따로 적용한다.
    probs = torch.softmax(logits, dim=1)[0]
    y_pred = int(torch.argmax(probs).item())

print("probs : ", probs.cpu().numpy())
print(f"실제값:{y_true}, 예측값:{y_pred}")

# 12) 시각화 - 학습 곡선
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="train acc")
plt.plot(history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

# 13) 혼동행렬 출력
all_preds = []
all_labels = []

loaded_model.eval()

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)

        outputs = loaded_model(x_batch)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())

cm = confusion_matrix(
    all_labels, all_preds, labels=list(range(10))
)

classes = [str(i) for i in range(10)]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=classes
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(
    ax=ax, cmap="Blues", values_format="d", colorbar=False
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()