실습2) LSTM으로 다음 숫자 예측

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

x = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10]
], dtype=np.float32)
y = np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float32)
x = x.reshape((8, 3, 1))    # 입력 형태(samples, time_steps, features)

# NumPy 배열을 PyTorch Tensor로 변환
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(x_tensor, y_tensor)

train_loader = DataLoader( dataset, batch_size=1, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 시계열(time_steps) 3개, features=1
        self.rnn = nn.LSTM(input_size=1, hidden_size=32, batch_first=True )
        # GRU는 LSTM보다 계산량이 적고 속도가 빠르다.
        # self.rnn = nn.GRU( input_size=1, hidden_size=32, batch_first=True )

        self.dense1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(16, 1)

    def forward(self, x):
        output, hidden = self.rnn(x)
        output = output[:, -1, :]  # 마지막 시점의 출력만 사용

        output = self.dense1(output)
        output = self.relu(output)
        output = self.dense2(output)
        return output

model = LSTMModel()
print(model)

criterion = nn.MSELoss()  # 평균제곱오차 손실 함수

# Adam 최적화 알고리즘
optimizer = torch.optim.Adam(model.parameters())
epochs = 1000
patience = 3

best_loss = float('inf')
wait = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion( pred, y_batch.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    print(
        f'Epoch {epoch + 1}/{epochs} - '
        f'loss: {epoch_loss:.6f}'
    )

    # 학습 손실이 개선되지 않으면 조기 종료
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        wait = 0
    else:
        wait += 1

        if wait >= patience:
            print('Early stopping')
            break

# 학습 데이터 예측
model.eval()

with torch.inference_mode():
    pred = model(x_tensor)

print('예측값 : ', pred.numpy().ravel())
print('실제값 : ', y.ravel())

# 새로운 값으로 예측
x_input = np.array([0, 1, 2], dtype=np.float32)
x_input = x_input.reshape((1, 3, 1))
x_input_tensor = torch.tensor( x_input, dtype=torch.float32)

with torch.inference_mode():
    new_pred = model(x_input_tensor)

print(new_pred.numpy().ravel())


