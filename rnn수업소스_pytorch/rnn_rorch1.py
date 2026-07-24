# RNN ...
# RNN은 순서가 있는 데이터를 다룬다.
# 이전 값들의 흐름을 기억하면서 다음 값을 예측한다.
# 노이즈가 있는 사인파 데이터를 이용해 RNN 예측 모델 작성

실습1)
import numpy as np
import matplotlib.pyplot as plt

xdata = np.linspace(-2 * np.pi, 2 * np.pi, 50)
# print(xdata)
sindata = np.sin(xdata) + 0.1 * np.random.randn(len(xdata))
sindata = sindata.astype(np.float32) 
print('sindata : ', sindata)

plt.plot(xdata, sindata)
plt.show()

n_rnn = 10
n_sample = len(xdata) - n_rnn    # 학습 샘플 개수

x = np.zeros((n_sample, n_rnn), dtype=np.float32)  # 입력 시퀀스를 저장할 배열
t = np.zeros((n_sample, n_rnn), dtype=np.float32)  # 예측 대상 시퀀스를 저장할 배열 (한 칸 뒤로 밀린 값을 정답 t로 생성)\

# 입력 시퀀스보다 한 칸 뒤의 값을 정답 시퀀스로 생성
for i in range(0, n_sample):
    x[i] = sindata[i:i + n_rnn]
    t[i] = sindata[i + 1:i + n_rnn + 1]

# PyTorch RNN 입력 형태 : (샘플 수, 시계열 수, 입력 특성 수)
x = x.reshape(n_sample, n_rnn, 1)
t = t.reshape(n_sample, n_rnn, 1)
print(x.shape)    # (40, 10, 1)
print(t.shape)    # (40, 10, 1)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# NumPy 배열을 PyTorch Tensor로 변환
x_tensor = torch.tensor(x, dtype=torch.float32)
t_tensor = torch.tensor(t, dtype=torch.float32)

# 전체 데이터 중 마지막 10%를 검증 데이터로 사용
train_size = int(n_sample * 0.9)

x_train = x_tensor[:train_size]
t_train = t_tensor[:train_size]

x_val = x_tensor[train_size:]
t_val = t_tensor[train_size:]

train_dataset = TensorDataset(x_train, t_train)
train_loader = DataLoader( train_dataset, batch_size=8, shuffle=True)

n_in = 1    # 입력층 뉴런 수
n_mid = 20  # 중간층 뉴런 수
n_out = 1   # 출력층 뉴런 수

# Simple RNN 모델 구축
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=n_in, hidden_size=n_mid, batch_first=True)
        self.dense = nn.Linear(n_mid, n_out)

    def forward(self, x):
        # output은 모든 시점의 RNN 출력 결과를 가진다.
        output, hidden = self.rnn(x)
        # 각 시점의 RNN 출력에 Linear 계층 적용
        output = self.dense(output)
        return output

model = RNNModel()
print(model)

criterion = nn.MSELoss()   # 평균제곱오차 손실 함수
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)  # SGD 최적화 알고리즘

epochs = 20
loss_list = []
val_loss_list = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, t_batch in train_loader:
        optimizer.zero_grad()    # 이전에 계산된 기울기 초기화
        output = model(x_batch)  # 예측
        loss = criterion(output, t_batch)  # 손실 계산
        loss.backward()   # 역전파
        optimizer.step()  # 가중치 수정

        total_loss += loss.item() * len(x_batch)

    train_loss = total_loss / len(train_dataset)
    loss_list.append(train_loss)

    # 검증 데이터 손실 계산
    model.eval()

    with torch.no_grad():
        val_output = model(x_val)
        val_loss = criterion(val_output, t_val).item()

    val_loss_list.append(val_loss)

    print(
        f'Epoch {epoch + 1:02d}, '
        f'loss: {train_loss:.5f}, '
        f'val_loss: {val_loss:.5f}'
    )

plt.plot(np.arange(len(loss_list)), loss_list) 
plt.plot(np.arange(len(val_loss_list)), val_loss_list)
plt.show()

# 예측
pred = x[0].reshape(-1)
print('pred : ', pred)

model.eval()

# 총 n_sample 횟수만큼 반복하며 예측값을 이어 붙이기
with torch.no_grad():
    for i in range(0, n_sample):
        input_data = pred[-n_rnn:].reshape(1, n_rnn, 1)

        input_tensor = torch.tensor(
            input_data, 
            dtype=torch.float32
        )

        yhat = model(input_tensor)

        # 출력의 마지막 시점 결과 추가
        pred = np.append(
            pred,
            yhat[0][n_rnn - 1][0].item()
        )

plt.plot(np.arange(len(sindata)), sindata, label='Train data')
plt.plot(np.arange(len(pred)), pred,label='Predicted')
plt.legend()
plt.show()

# 예측값과 실제값 비교
predicted = pred[n_rnn:]
actual = sindata[n_rnn:]

for i in range(10):
    print(
        f'{i:02d} - '
        f'예측:{predicted[i]:.4f}, '
        f'실제:{actual[i]:.4f}'
    )

# MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual, predicted)
print(f'Mse : {mse:.5f}')
