# 딥러닝으로 이진분류 : 전통적인 Logistic Regression을 신경망 구조로 확장한 예제
# Logistic Regression:
#   z = w1*x1 + w2*x2 + b
#   sigmoid(z) -> 0~1 사이 확률 출력
# 신경망 이진분류:
#   입력층 -> 은닉층(ReLU) -> 출력층(Sigmoid)
#   출력값은 1일 확률로 해석
# PyTorch에서는 TensorFlow의 model.fit() 대신
# forward -> loss 계산 -> backward -> optimizer.step() 흐름을 직접 작성한다.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 랜덤 고정
np.random.seed(42)
torch.manual_seed(42)

# 1. 데이터 생성
x_data = np.array([[1, 2],[2, 3],[3, 4],[4, 3],[3, 2],[2, 1]],dtype=np.float32)
y_data = np.array([[0],[0],[0],[1],[1],[1]],dtype=np.float32)

# PyTorch Tensor로 변환
x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

# 공통 학습 함수
def train_model(model, x_train, y_train, epochs=20, lr=0.01, batch_size=1):
    """
    학습 흐름: 1. 예측값 계산 -> 2. 손실 계산 -> 3. gradient 초기화
                -> 4. 역전파 -> 5. 파라미터 갱신
    """
    # Binary Cross Entropy : 출력층에서 Sigmoid를 사용했기 때문에 BCELoss 사용
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data_size = len(x_train)

    for epoch in range(epochs):
        indices = torch.randperm(data_size) # 매 epoch마다 데이터 순서 섞기
        total_loss = 0

        for i in range(0, data_size, batch_size):
            batch_idx = indices[i:i + batch_size]

            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            pred = model(x_batch)  # 1) forward: 예측 확률 계산
            loss = criterion(pred, y_batch)  # 2) 손실 계산
            optimizer.zero_grad()  # 3) 이전 gradient 초기화
            loss.backward()   # 4) 역전파
            optimizer.step()  # 5) 파라미터 업데이트
            total_loss += loss.item()
    return model


# 공통 평가 함수
def evaluate_model(model, x, y):
    criterion = nn.BCELoss()

    # 평가 시에는 gradient 계산이 필요 없음
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y).item()

        # 확률값을 0 또는 1로 변환
        pred_class = (pred >= 0.5).float()

        accuracy = (pred_class == y).float().mean().item()

    return loss, accuracy


# 1) nn.Sequential 버전
print('1) nn.Sequential 버전 (빠른 구현)')
# TensorFlow Sequential API와 가장 비슷한 방식
# 층을 순서대로 쌓는 단순 구조
# 입력 2개 -> 은닉층 4개 -> 출력 1개
# 마지막 Sigmoid는 이진분류 확률 출력을 위해 사용
seq_model = nn.Sequential(
    nn.Linear(in_features=2, out_features=4),
    nn.ReLU(),
    nn.Linear(in_features=4, out_features=1),
    nn.Sigmoid()
)
print(seq_model)

# 모델 학습
seq_model = train_model(
    model=seq_model,
    x_train=x_tensor, y_train=y_tensor, epochs=20, lr=0.01, batch_size=1
)

# 모델 평가
loss, acc = evaluate_model(seq_model, x_tensor, y_tensor)
print([loss, acc])
print(f'평가 결과 : 손실={loss:.4f}, 정확도={acc:.4f}')


# 예측값과 실제값으로 시각화
# 2차원 입력(x1, x2)을 가진 모델을
# x2는 고정하고 x1만 변화시키면서 sigmoid 형태의 출력 변화를 확인
x1_range = np.linspace(0, 6, 100).astype(np.float32)
x2_fixed = 2.5

# x1_range와 같은 길이로 x2 값을 2.5로 고정
x_vis = np.column_stack([
    x1_range, np.full_like(x1_range, x2_fixed)
]).astype(np.float32)

x_vis_tensor = torch.tensor(x_vis, dtype=torch.float32)

# 예측 확률 계산
with torch.no_grad():
    y_prob = seq_model(x_vis_tensor).numpy()

x1_real = x_data[:, 0]
y_real = y_data.ravel()

plt.figure(figsize=(7, 5))
plt.plot(x1_range, y_prob, label='sigmoid curve')
plt.scatter(x1_real, y_real, color='red', label='True data')
plt.xlabel('x data')
plt.ylabel('probability')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# 정확도 계산
with torch.no_grad():
    pred = seq_model(x_tensor).numpy()

pred_class = (pred >= 0.5).astype(int)

accuracy = accuracy_score(y_data, pred_class)
print(f'1) 정확도 : {accuracy:.4f}')

# 새로운 값으로 분류 예측
new_data = np.array([[1, 2], [10, 5]], dtype=np.float32)
new_tensor = torch.tensor(new_data, dtype=torch.float32)
with torch.no_grad():
    pred = seq_model(new_tensor).numpy()

print('예측 확률 : ', pred.ravel())
print('예측 결과:', (pred >= 0.5).astype(int).ravel())
print('예측 결과:', [1 if i >= 0.5 else 0 for i in pred])
print('예측 결과:', np.where(pred >= 0.5, 1, 0).ravel())


# 2) nn.Module 클래스 버전 -------------
print('\n2) nn.Module 클래스 버전 - 가장 일반적인 방법임')
# TensorFlow Functional API처럼 구조를 명확하게 정의하고 싶을 때
# PyTorch에서는 nn.Module 클래스를 상속하여 모델을 만든다. 
class FunctionalLikeModel(nn.Module):
    def __init__(self):    # 사용할 layer 정의
        super().__init__()

        self.dense1 = nn.Linear(in_features=2, out_features=4)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=4, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):  # 데이터가 흘러가는 연산 순서 정의
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

model_func = FunctionalLikeModel()
print(model_func)

model_func = train_model(
    model=model_func,
    x_train=x_tensor, y_train=y_tensor, epochs=20, lr=0.01, batch_size=1
)

loss2, acc2 = evaluate_model(model_func, x_tensor, y_tensor)
print([loss2, acc2])
print(f'평가 결과2 : 손실={loss2:.4f}, 정확도={acc2:.4f}')


# 3) 다중 입력 모델 ------------------------------------
print('\n3) nn.Module 버전 - 다중 입력')
# TensorFlow Functional API의 다중 입력 구조와 비슷한 예제
# 기존: [x1, x2] -> Dense -> Dense -> 출력
# 다중 입력:
#   x1 -> Linear
#                 -> concat -> Linear -> Sigmoid -> 출력
#   x2 -> Linear
#
# PyTorch에서는 forward(self, input1, input2)처럼 여러 개의 입력을 직접 받을 수 있다.
class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()

        # x1 입력을 따로 처리
        self.x1_layer = nn.Linear(in_features=1, out_features=2)

        # x2 입력을 따로 처리
        self.x2_layer = nn.Linear(in_features=1, out_features=4)
        self.relu = nn.ReLU()

        # x1 처리 결과 2개 + x2 처리 결과 4개 = 총 6개 입력
        self.output_layer = nn.Linear(in_features=6, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        x1 = self.x1_layer(input1)
        x1 = self.relu(x1)
        x2 = self.x2_layer(input2)
        x2 = self.relu(x2)
        # dim=1 방향으로 feature 결합
        merged = torch.cat([x1, x2], dim=1)

        output = self.output_layer(merged)
        output = self.sigmoid(output)

        return output


# x_data를 x1, x2로 분리
x1_data = x_data[:, 0].reshape(-1, 1).astype(np.float32)
x2_data = x_data[:, 1].reshape(-1, 1).astype(np.float32)

x1_tensor = torch.tensor(x1_data, dtype=torch.float32)
x2_tensor = torch.tensor(x2_data, dtype=torch.float32)

multi_model = MultiInputModel()
print(multi_model)


def train_multi_input_model(model, x1_train, x2_train, y_train, epochs=20, lr=0.01, batch_size=1):
    """
    다중 입력 모델 학습 함수
    입력이 하나가 아니라 input1, input2 두 개이므로
    model(x1_batch, x2_batch) 형태로 호출한다.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data_size = len(y_train)

    for epoch in range(epochs):
        indices = torch.randperm(data_size)

        for i in range(0, data_size, batch_size):
            batch_idx = indices[i:i + batch_size]

            x1_batch = x1_train[batch_idx]
            x2_batch = x2_train[batch_idx]
            y_batch = y_train[batch_idx]

            # 다중 입력 forward
            pred = model(x1_batch, x2_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_multi_input_model(model, x1, x2, y):
    criterion = nn.BCELoss()

    with torch.no_grad():
        pred = model(x1, x2)
        loss = criterion(pred, y).item()
        pred_class = (pred >= 0.5).float()
        accuracy = (pred_class == y).float().mean().item()

    return loss, accuracy


multi_model = train_multi_input_model(
    model=multi_model,
    x1_train=x1_tensor, x2_train=x2_tensor, y_train=y_tensor, epochs=20,
    lr=0.01, batch_size=1
)

loss_multi, acc_multi = evaluate_multi_input_model(
    model=multi_model, x1=x1_tensor, x2=x2_tensor, y=y_tensor
)

print([loss_multi, acc_multi])
print(f'평가 결과 multi : 손실={loss_multi:.4f}, 정확도={acc_multi:.4f}')


# 4) Model Subclassing 방식
print('\n4) nn.Module Subclassing 방식')
# TensorFlow의 Model Subclassing과 가장 비슷한 방식
# PyTorch 모델은 기본적으로 nn.Module을 상속해서 만든다.
# 따라서 PyTorch에서는 이 방식이 가장 일반적이고 실무적인 방식이다.
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 레이어 정의
        self.dense1 = nn.Linear(in_features=2, out_features=4)
        self.dense2 = nn.Linear(in_features=4, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # forward 연산 정의
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

sub_model = MyModel()
print(sub_model)

sub_model = train_model(
    model=sub_model,
    x_train=x_tensor, y_train=y_tensor, epochs=20,lr=0.01,batch_size=1
)

loss_sub, acc_sub = evaluate_model(sub_model, x_tensor, y_tensor)
print([loss_sub, acc_sub])
print(f'평가 결과 sub : 손실={loss_sub:.4f}, 정확도={acc_sub:.4f}')