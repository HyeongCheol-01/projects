# 이항분류(sigmoid)는 다항분류(softmax)로도 처리 가능하다.
# PyTorch 버전
# 1. 이항분류 방식 : 출력층: Linear(..., 1), 손실함수: BCEWithLogitsLoss()
# 2. 다항분류 방식 : 출력층: Linear(..., 2), 손실함수: CrossEntropyLoss()
# PyTorch에서는 학습 시 sigmoid, softmax를 모델에 직접 넣지 않는 것이 일반적이다.
# 예측할 때만 sigmoid 또는 softmax를 적용한다.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(123)
torch.manual_seed(123)

# 2. 데이터 읽기
datas = np.loadtxt(
    'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/diabetes.csv',
    delimiter=','
)

print(datas.shape)   # (759, 9)
print(datas[:1])
print(set(datas[:, -1]))

# 3. feature / label 분리
x = datas[:, 0:8].astype(np.float32)
y = datas[:, -1].astype(np.float32)
print(x.shape)
print(y.shape)

# 4. train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123, stratify=y)
print(x_train.shape, x_test.shape)  # (531, 8) (228, 8)

# 5. Scaling
# 딥러닝 학습 안정성을 위해 표준화 적용 : train 데이터로 fit, test 데이터는 transform만 수행
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
x_test_scaled = scaler.transform(x_test).astype(np.float32)

# 6. Tensor 변환
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)

# 이항분류 BCEWithLogitsLoss용 label.  shape을 (N, 1)로 맞춤
y_train_binary_tensor = torch.tensor( y_train.reshape(-1, 1), dtype=torch.float32 )
y_test_binary_tensor = torch.tensor( y_test.reshape(-1, 1), dtype=torch.float32 )

# 다항분류 CrossEntropyLoss용 label. 원핫인코딩하지 않고 정수 label 그대로 사용. dtype은 반드시 long
y_train_multi_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_multi_tensor = torch.tensor(y_test, dtype=torch.long)

# 7. 이항분류 모델 정의
class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# 8. 다항분류 모델 정의 : 이항분류 문제지만 class를 2개로 보고 softmax 다항분류처럼 처리
# CrossEntropyLoss가 내부적으로 LogSoftmax를 처리함
class SoftmaxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# 9. 이항분류 학습 함수
def train_binary_model(model, x_train, y_train, epochs=100, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = { "loss": [], "acc": [] }

    for epoch in range(epochs):
        model.train()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy 계산
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            acc = (pred == y_train).float().mean().item()

        history["loss"].append(loss.item())
        history["acc"].append(acc)

    return history


# 10. 이항분류 평가 함수
def evaluate_binary_model(model, x_test, y_test):
    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    with torch.no_grad():
        logits = model(x_test)
        loss = criterion(logits, y_test).item()
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        acc = (pred == y_test).float().mean().item()

    return loss, acc


# 11. 다항분류 학습 함수
def train_softmax_model(model, x_train, y_train, epochs=100, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = { "loss": [], "acc": [] }

    for epoch in range(epochs):
        model.train()

        # 1) forward.  logits shape: (N, 2)
        logits = model(x_train)

        # 2) loss 계산 : CrossEntropyLoss는 softmax 전의 logits와 정수 label을 입력으로 받음
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accuracy 계산
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y_train).float().mean().item()

        history["loss"].append(loss.item())
        history["acc"].append(acc)

    return history


# 12. 다항분류 평가 함수
def evaluate_softmax_model(model, x_test, y_test):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        logits = model(x_test)
        loss = criterion(logits, y_test).item()

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y_test).float().mean().item()

    return loss, acc


print('\n이항분류(sigmoid 방식)')
binary_model = BinaryModel()

binary_history = train_binary_model(
    model=binary_model, x_train=x_train_tensor, y_train=y_train_binary_tensor,
    epochs=100, lr=0.001
)

binary_scores = evaluate_binary_model(
    model=binary_model, x_test=x_test_tensor, y_test=y_test_binary_tensor
)

print('sigmoid scores : ', binary_scores)


print('\n다항분류(softmax 방식)')
softmax_model = SoftmaxModel()

softmax_history = train_softmax_model(
    model=softmax_model, x_train=x_train_tensor, y_train=y_train_multi_tensor,
    epochs=100, lr=0.001
)

softmax_scores = evaluate_softmax_model(
    model=softmax_model, x_test=x_test_tensor, y_test=y_test_multi_tensor
)
print('softmax scores : ', softmax_scores)

print('\n예측 결과 비교')

with torch.no_grad():
    # 이항분류 모델 예측
    binary_logits = binary_model(x_test_tensor[:5])
    binary_prob = torch.sigmoid(binary_logits)
    binary_pred = (binary_prob >= 0.5).int().ravel()

    # 다항분류 모델 예측
    softmax_logits = softmax_model(x_test_tensor[:5])
    softmax_prob = torch.softmax(softmax_logits, dim=1)
    softmax_pred = torch.argmax(softmax_prob, dim=1)

print('이항분류 예측 확률 : ', binary_prob.numpy().ravel())
print('이항분류 예측값 : ', binary_pred.numpy())
print('\n다항분류 예측 확률 : ', softmax_prob.numpy())
print('다항분류 예측값 : ', softmax_pred.numpy())
print('\n실제값 : ', y_test[:5].astype(int))