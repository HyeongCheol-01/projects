# 동물의 타입 분류
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 데이터 읽기
datas = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/zoo.csv"
)
print(datas.head(3))
print(datas.info())

# 2. 입력 데이터 / 정답 데이터 분리
# 마지막 열은 class_type, 나머지는 입력 feature
x_data = datas.iloc[:, :-1].astype("float32").values
y_data = datas.iloc[:, -1].astype("int64").values
print("x 첫 번째 데이터:", x_data[0])
print("x shape:", x_data.shape)
print("원본 y 첫 번째 라벨:", y_data[0])
print("원본 y 라벨 종류:", sorted(set(map(int, y_data))))

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 4. 라벨 매핑
# CrossEntropyLoss는 라벨이 반드시 0부터 시작해야 함.
# 예) 원본 라벨이 [1, 2, 3, 4, 5, 6, 7]이면
# 학습용 라벨은 [0, 1, 2, 3, 4, 5, 6]으로 변환
# 원본 라벨이 이미 [0, 1, 2, 3, 4, 5, 6]이어도 문제 없음.
original_labels = sorted(set(y_data))

label_to_index = {
    label: idx for idx, label in enumerate(original_labels)
}
index_to_label = {
    idx: label for label, idx in label_to_index.items()
}
y_data_index = np.array(
    [label_to_index[label] for label in y_data], dtype=np.int64
)
nb_classes = len(original_labels)
print("원본 라벨 목록:", original_labels)
print("라벨 -> 인덱스 매핑:", label_to_index)
print("인덱스 -> 라벨 매핑:", index_to_label)
print("학습용 라벨 종류:", sorted(set(map(int, y_data_index))))
print("클래스 개수:", nb_classes)

# 5. train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data_index, test_size=0.2, random_state=42, stratify=y_data_index
)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# 6. NumPy 배열을 PyTorch Tensor로 변환
# 입력 x: float32, 정답 y: long
# CrossEntropyLoss는 y가 torch.long 타입이어야 함.
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 7. DataLoader 생성
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 8. PyTorch 모델 정의
class ZooModel(nn.Module):
    def __init__(self, input_dim, nb_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(32, nb_classes)
        )

    def forward(self, x):
        return self.net(x)

model = ZooModel(
    input_dim=x_train.shape[1], nb_classes=nb_classes).to(device)
print(model)

# 9. 손실함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10. 학습
epochs = 50
train_loss_history = []
train_acc_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)   # 1. 예측
        loss = criterion(outputs, batch_y)  # 2. 손실 계산
        optimizer.zero_grad()   # 3. 기존 gradient 초기화
        loss.backward()    # 4. 역전파
        optimizer.step()   # 5. 가중치 갱신
        total_loss += loss.item() * batch_x.size(0)  # 손실 누적

        # 정확도 계산
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total

    train_loss_history.append(avg_loss)
    train_acc_history.append(avg_acc)

    print(
        f"Epoch [{epoch + 1:02d}/{epochs}] "
        f"loss: {avg_loss:.4f}, acc: {avg_acc:.4f}"
    )

# 11. 테스트 데이터 평가
model.eval()

with torch.no_grad():
    x_test_device = x_test_tensor.to(device)
    y_test_device = y_test_tensor.to(device)

    test_outputs = model(x_test_device)
    test_loss = criterion(test_outputs, y_test_device)
    test_pred = torch.argmax(test_outputs, dim=1)
    test_acc = (test_pred == y_test_device).float().mean()

print(f"\nloss:{test_loss.item():.4f}, acc:{test_acc.item():.4f}")


# 12. 학습 과정 시각화
plt.plot(train_loss_history, label="train loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(train_acc_history, label="train accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# 13. 혼동행렬 및 분류 리포트
y_pred_index = test_pred.cpu().numpy()
y_true_index = y_test_tensor.numpy()

# 학습용 라벨 index를 원래 라벨로 복원
y_pred_original = np.array(
    [index_to_label[idx] for idx in y_pred_index]
)
y_true_original = np.array(
    [index_to_label[idx] for idx in y_true_index]
)
print("\nclassification_report : \n")
print(classification_report(y_true_original, y_pred_original))

cm = confusion_matrix(y_true_original, y_pred_original)
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("true")
plt.show()

print("\n새로운 값으로 분류 예측")
new_data = np.array(
    [[0., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 0., 0., 4., 0., 0., 1.]], dtype="float32"
)
new_tensor = torch.tensor(new_data, dtype=torch.float32).to(device)
model.eval()

with torch.no_grad():
    logits = model(new_tensor)
    # 모델 출력 logits를 확률로 변환
    probs = torch.softmax(logits, dim=1)
    # 가장 확률이 높은 클래스 index
    pred_index = torch.argmax(probs, dim=1).item()
    # 학습용 index를 원래 class_type 라벨로 복원
    pred_label = index_to_label[pred_index]

print("분류 예측 확률:", probs.cpu().numpy().ravel())
print("분류 예측 index:", pred_index)
print("분류 예측 라벨:", pred_label)