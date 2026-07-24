# iris dataset으로 꽃 종류 분류기
# ROC Curve까지 표현
# layer 수에 따른 모델 성능 비교

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 데이터 읽기
iris = load_iris()
print(iris.keys())
x = iris.data          # feature
y = iris.target        # label
print(x[:2])
print(y[:2])

names = iris.target_names
print(names)  # ['setosa' 'versicolor' 'virginica']
feature_names = iris.feature_names
print(feature_names)  # ['sepal length (cm)', ...]

# 2. 정답 라벨 확인
# 원핫 처리 : CrossEntropyLoss는 정수 라벨을 그대로 사용한다.
# 예) y = 0, 1, 2  따라서 PyTorch에서는 원핫 인코딩하지 않는다.
print("y shape:", y.shape)
print("y label 종류:", sorted(set(y)))

# 3. feature 표준화
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print("표준화 전:")
print(x[:2])
print("표준화 후:")
print(x_scale[:2])

# 4. train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_scale, y, test_size=0.3, random_state=42, stratify=y
)
print(x_train.shape, x_test.shape)  # (105, 4) (45, 4)
n_features = x_train.shape[1]
n_classes = len(set(y_train))
print("입력 feature 수:", n_features)
print("출력 class 수:", n_classes)

# 5. 랜덤 시드 고정
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 6. NumPy 배열을 PyTorch Tensor로 변환
# x: float32, y: long
# CrossEntropyLoss는 정답 라벨이 torch.long 타입이어야 한다.
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 7. train 데이터를 train / validation으로 분리
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train_tensor, y_train_tensor, test_size=0.3,
    random_state=42, stratify=y_train_tensor
)
print("실제 학습 데이터:", x_tr.shape)
print("검증 데이터:", x_val.shape)

# 8. DataLoader 생성 : DataLoader는 데이터를 batch 단위로 공급한다.
batch_size = 4
train_dataset = TensorDataset(x_tr, y_tr)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# 9. layer 수가 다른 모델 생성 클래스
class IrisModel(nn.Module):
    def __init__(self, input_dim, output_dim, out_nodes, n_hidden_layers, model_name="model"):
        super().__init__()

        self.model_name = model_name

        layers = []

        # 첫 번째 hidden layer
        layers.append(nn.Linear(input_dim, out_nodes))
        layers.append(nn.ReLU())

        # 추가 hidden layer
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(out_nodes, out_nodes))
            layers.append(nn.ReLU())

        # 출력 layer : softmax를 넣지 않고 logits 출력
        layers.append(nn.Linear(out_nodes, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 10. 모델 구조 확인
models = [
    IrisModel(
        input_dim=n_features, output_dim=n_classes,
        out_nodes=10, n_hidden_layers=n, model_name=f"model_{n}"
    ).to(device) for n in range(1, 4)
]

for model in models:
    print()
    print("모델명:", model.model_name)
    print(model)

# 11. 학습 함수 정의
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "loss": [], "acc": [], "val_loss": [], "val_acc": []
    }

    for epoch in range(epochs):
        # 학습 모드
        model.train()

        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)   # 1. 예측
            loss = criterion(outputs, batch_y)  # 2. 손실 계산
            optimizer.zero_grad()   # 3. 기존 gradient 초기화
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item() * batch_x.size(0)

            # 정확도 계산
            pred = torch.argmax(outputs, dim=1)
            train_correct += (pred == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_avg_loss = train_total_loss / train_total
        train_avg_acc = train_correct / train_total

        # 검증 모드
        model.eval()

        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_total_loss += loss.item() * batch_x.size(0)

                pred = torch.argmax(outputs, dim=1)
                val_correct += (pred == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_avg_loss = val_total_loss / val_total
        val_avg_acc = val_correct / val_total

        history["loss"].append(train_avg_loss)
        history["acc"].append(train_avg_acc)
        history["val_loss"].append(val_avg_loss)
        history["val_acc"].append(val_avg_acc)

    return history

# 12. 테스트 평가 함수 정의
def evaluate_model(model, x_test_tensor, y_test_tensor):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        x_test_device = x_test_tensor.to(device)
        y_test_device = y_test_tensor.to(device)

        outputs = model(x_test_device)
        loss = criterion(outputs, y_test_device)

        pred = torch.argmax(outputs, dim=1)
        acc = (pred == y_test_device).float().mean()

    return loss.item(), acc.item()

# 13. 모델별 학습 및 평가
history_dict = {}
epochs = 50

for model in models:
    print()
    print("모델명:", model.model_name)

    history = train_model(
        model=model, train_loader=train_loader,
        val_loader=val_loader, epochs=epochs, lr=0.001
    )

    loss, acc = evaluate_model(model, x_test_tensor, y_test_tensor)
    print(f"loss:{loss:.4f}, acc:{acc:.4f}")
    history_dict[model.model_name] = [history, model]

print(history_dict.keys())

# 14. validation accuracy / validation loss 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

for model_name in history_dict:
    history = history_dict[model_name][0]

    val_acc = history["val_acc"]
    val_loss = history["val_loss"]

    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)

ax1.set_ylabel("val acc")
ax2.set_ylabel("val loss")
ax2.set_xlabel("epoch")
ax1.legend()
ax2.legend()
plt.show()

# 15. ROC Curve
# ROC Curve는 일반적으로 이진분류에서 많이 사용하지만,
# 다중분류에서는 One-vs-Rest 방식으로 확장해서 볼 수 있다.
# 여기서는 기존 TensorFlow 코드와 동일하게
# y_test 원핫 형태와 y_pred 확률을 ravel()로 펼쳐서
# micro-average 방식에 가까운 ROC Curve를 그린다.
# PyTorch 모델은 logits를 출력하므로 ROC Curve에는 softmax 확률값을 사용한다.

def to_onehot(y, num_classes):
    return np.eye(num_classes)[y]

y_test_onehot = to_onehot(y_test, n_classes)

plt.figure()
plt.plot([0, 1], [0, 1], "k--")

for model_name in history_dict:
    model = history_dict[model_name][1]
    model.eval()

    with torch.no_grad():
        x_test_device = x_test_tensor.to(device)
        logits = model(x_test_device)
        # logits를 확률로 변환
        y_pred_prob = torch.softmax(logits, dim=1)
        y_pred_prob = y_pred_prob.cpu().numpy()

    fpr, tpr, _ = roc_curve(
        y_test_onehot.ravel(),
        y_pred_prob.ravel()
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr, label=f"{model_name}, AUC:{roc_auc:.3f}"
    )

plt.xlabel("fpr(false positive rate)")
plt.ylabel("tpr(true positive rate)")
plt.title("ROC Curve")
plt.legend()
plt.show()