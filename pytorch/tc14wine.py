# 와인의 등급, 맛, 산도 등의 정보를 이용해
# 레드 와인 / 화이트 와인을 분류하는 이진분류 모델
# label: 0 또는 1
# PyTorch 이진분류 권장 방식:
#   모델 출력층에는 sigmoid를 넣지 않음
#   손실 함수는 nn.BCEWithLogitsLoss() 사용
#   예측할 때만 torch.sigmoid() 적용

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 랜덤 고정
np.random.seed(12)
torch.manual_seed(12)

# GPU 사용 가능하면 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 2. 데이터 읽기
wdf = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/wine.csv",
    header=None
)

print(wdf.head(2))
print(wdf.info())

# 마지막 컬럼이 label
print(wdf.iloc[:, 12].unique())        # [1 0]
print(len(wdf[wdf.iloc[:, 12] == 0]))  # 4898
print(len(wdf[wdf.iloc[:, 12] == 1]))  # 1599

# 3. feature / label 분리
# 0 ~ 11번 컬럼 : 입력 feature,  12번 컬럼 : label
dataset = wdf.values
x = dataset[:, 0:12].astype(np.float32)
y = dataset[:, -1].astype(np.float32)
print(x[:2])
print(y[:2])

# 4. train / test split
# stratify=y: label 0과 1의 비율을 train/test에 비슷하게 유지
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=12, stratify=y, shuffle=True
)
print(x_train[:2], x_train.shape)  # (4547, 12)
print(y_train[:2], y_train.shape)  # (4547,)

# 5. 스케일링
# 와인 데이터는 feature별 값의 범위가 다르기 때문에 StandardScaler를 적용하면 학습이 더 안정적임
# 중요: scaler는 train 데이터로 fit, test 데이터는 transform만 수행
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
x_test_scaled = scaler.transform(x_test).astype(np.float32)

# 6. train 데이터를 다시 train / validation으로 분리
# TensorFlow의 validation_split=0.2와 비슷한 역할
x_train_part, x_val, y_train_part, y_val = train_test_split(
    x_train_scaled, y_train, test_size=0.2,
    random_state=12, stratify=y_train
)

print("train:", x_train_part.shape, y_train_part.shape)
print("val  :", x_val.shape, y_val.shape)
print("test :", x_test_scaled.shape, y_test.shape)

# 7. NumPy 배열을 PyTorch Tensor로 변환
# BCEWithLogitsLoss는 예측값과 정답의 shape이 같아야 함
# y는 (데이터수,)가 아니라 (데이터수, 1) 형태로 변환
x_train_tensor = torch.tensor(x_train_part, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_part.reshape(-1, 1), dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# 8. DataLoader 생성 : TensorFlow의 batch_size=64와 같은 역할
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 9. PyTorch 모델 정의
# BCEWithLogitsLoss가 sigmoid + BCE를 내부적으로 안정적으로 처리함
class WineClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),

            nn.Linear(24, 12),
            nn.ReLU(),

            nn.Linear(12, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = WineClassifier().to(device)
print(model)

# 10. 손실 함수와 optimizer 정의
#   criterion = nn.BCEWithLogitsLoss(), optimizer = optim.Adam(...)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 11. 평가 함수 정의
def evaluate_model(model, x_data, y_data):
    """
    PyTorch 평가 함수
    model.eval(): 평가 모드
        Dropout, BatchNorm 등이 학습과 다르게 동작하도록 설정
    torch.no_grad(): 평가 중에는 gradient 계산이 필요 없으므로 메모리 절약
    """

    model.eval()
    x_data = x_data.to(device)
    y_data = y_data.to(device)

    with torch.no_grad():
        logits = model(x_data)
        loss = criterion(logits, y_data).item()

        # logits -> sigmoid -> 확률
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        acc = (pred == y_data).float().mean().item()

    return loss, acc


# 12. 훈련되지 않은 모델의 정확도
train_loss_before, train_acc_before = evaluate_model(
    model, x_train_tensor, y_train_tensor
)
print(f"훈련되지 않은 모델의 정확도: {train_acc_before * 100:.2f}%")

# 13. EarlyStopping + ModelCheckpoint 설정
# TensorFlow EarlyStopping: PyTorch에서는 직접 구현
patience = 5
best_val_loss = np.inf
patience_count = 0

MODEL_DIR = "./winemodel/"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

modelpath = "./winemodel/winemodel_pytorch.pth"

# 14. 모델 학습
epochs = 1000

history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoch in range(epochs):
    model.train()

    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)  # 1) 예측값 계산
        loss = criterion(logits, batch_y)  # 2) 손실 계산
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # batch loss 누적
        train_loss_sum += loss.item() * batch_x.size(0)

        prob = torch.sigmoid(logits)  # accuracy 계산
        pred = (prob >= 0.5).float()

        train_correct += (pred == batch_y).sum().item()
        train_total += batch_y.size(0)

    # epoch 단위 train loss / accuracy
    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # validation 평가
    val_loss, val_acc = evaluate_model(
        model, x_val_tensor, y_val_tensor
    )

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"loss: {train_loss:.4f} "
        f"accuracy: {train_acc:.4f} "
        f"val_loss: {val_loss:.4f} "
        f"val_accuracy: {val_acc:.4f}"
    )

    # ModelCheckpoint : validation loss가 가장 낮을 때 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_count = 0

        torch.save(model.state_dict(), modelpath)
        print(f"  -> 모델 저장 완료: {modelpath}")
    else:
        patience_count += 1

    # EarlyStopping
    # patience 횟수만큼 val_loss가 개선되지 않으면 학습 중단
    if patience_count >= patience:
        print(f"\nEarlyStopping 발생: {epoch + 1} epoch에서 학습 중단")
        break

# 15. 저장된 best model 불러오기
best_model = WineClassifier().to(device)
best_model.load_state_dict(torch.load(modelpath, map_location=device))
best_model.eval()

# 16. 테스트 데이터 평가
test_loss, test_acc = evaluate_model(
    best_model, x_test_tensor, y_test_tensor
)
print(f"\n훈련된 모델의 정확도: {test_acc * 100:.2f}%")
print(f"테스트 손실: {test_loss:.4f}")

# 17. 시각화 - loss
epoch_len = np.arange(len(history["loss"]))

plt.figure(figsize=(8, 5))
plt.plot(epoch_len, history["val_loss"], c="red", label="val_loss")
plt.plot(epoch_len, history["loss"], c="blue", label="loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.show()

# 18. 시각화 - accuracy
plt.figure(figsize=(8, 5))
plt.plot(epoch_len, history["val_accuracy"], c="red", label="val_accuracy")
plt.plot(epoch_len, history["accuracy"], c="blue", label="accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 19. 저장된 모델로 예측
# PyTorch: 모델 구조 생성
#   load_state_dict()로 저장된 가중치 불러오기
#   torch.sigmoid()로 확률 변환

new_data = x_test[:5, :]
print("\n원본 new_data")
print(new_data)

# 새로운 데이터도 반드시 학습 때 사용한 scaler로 변환해야 함
new_data_scaled = scaler.transform(new_data).astype(np.float32)

new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    logits = best_model(new_data_tensor)
    new_prob = torch.sigmoid(logits)  # logits를 확률로 변환
    new_pred = (new_prob >= 0.5).int() # 0.5 기준으로 0 또는 1 분류

print("\n예측 확률:", new_prob.cpu().numpy().ravel())
print("예측 결과:", new_pred.cpu().numpy().ravel())
print("실제 정답:", y_test[:5].astype(int))