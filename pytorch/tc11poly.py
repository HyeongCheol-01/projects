# 다항회귀 : 데이터가 비선형 분포인 경우
# 회귀선이 2차, 3차 함수 등의 곡선 형태가 될 수 있음
# 다항회귀는 하나의 feature에 대해 x, x^2, x^3 ... 처럼 차수를 확장하여
# 선형 모델이 비선형 관계를 학습할 수 있도록 입력 feature를 변환하는 방식

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 랜덤 고정
np.random.seed(7)
torch.manual_seed(7)

# 가상의 feature와 label 만들기
x = np.linspace(-3, 3, 40).reshape(-1, 1).astype(np.float32)
print(x[:3])

# y = x^2 + x + 2 + noise
y = (
    (x[:, 0] ** 2)
    + x[:, 0]
    + 2
    + np.random.normal(0, 1.5, size=len(x))
).astype(np.float32)

print(y[:3])

# PyTorch Tensor로 변환
# x_tensor shape : (40, 1)
# y_tensor shape : (40, 1)
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# R2 score 계산 함수
def r2_score_np(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)          # 잔차 제곱합
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # 전체 제곱합

    return 1 - (ss_res / ss_tot)


# -------------------------------------------------------
# 1. 선형회귀 모델
# -------------------------------------------------------
# TensorFlow의 Dense(units=1)과 동일한 역할
# 입력 feature 1개 → 출력 1개
linear_model = nn.Linear(in_features=1, out_features=1)

# 손실 함수 : 평균제곱오차 MSE
criterion = nn.MSELoss()

# 옵티마이저 : Adam
linear_optimizer = optim.Adam(linear_model.parameters(), lr=0.05)

# 학습
epochs = 500

for epoch in range(epochs):
    # 1) 예측
    y_pred = linear_model(x_tensor)

    # 2) 손실 계산
    loss = criterion(y_pred, y_tensor)

    # 3) 이전 gradient 초기화
    linear_optimizer.zero_grad()

    # 4) 역전파
    loss.backward()

    # 5) 파라미터 업데이트
    linear_optimizer.step()

# 학습 데이터에 대한 예측
with torch.no_grad():
    y_pred_linear = linear_model(x_tensor).numpy()


# -------------------------------------------------------
# 2. 다항회귀
# -------------------------------------------------------
# x_poly = [x, x^2]
# 하나의 feature x를 x와 x^2 두 개의 feature로 확장
x_poly = np.column_stack([
    x[:, 0],
    x[:, 0] ** 2
]).astype(np.float32)

print(x_poly[:3])

# PyTorch Tensor로 변환
# x_poly_tensor shape : (40, 2)
x_poly_tensor = torch.tensor(x_poly, dtype=torch.float32)

# 다항회귀 모델
# 입력 feature 2개(x, x^2) → 출력 1개
# 모델 자체는 선형 모델이지만, 입력에 x^2이 포함되어 있으므로 곡선 형태를 학습할 수 있음
poly_model = nn.Linear(in_features=2, out_features=1)

poly_optimizer = optim.Adam(poly_model.parameters(), lr=0.05)

# 학습
for epoch in range(epochs):
    # 1) 예측
    y_pred = poly_model(x_poly_tensor)

    # 2) 손실 계산
    loss = criterion(y_pred, y_tensor)

    # 3) gradient 초기화
    poly_optimizer.zero_grad()

    # 4) 역전파
    loss.backward()

    # 5) 파라미터 업데이트
    poly_optimizer.step()

# 학습 데이터에 대한 예측
with torch.no_grad():
    y_pred_poly = poly_model(x_poly_tensor).numpy()


# -------------------------------------------------------
# 3. 그래프용 예측 데이터 만들기
# -------------------------------------------------------
# 부드러운 곡선을 그리기 위한 x축 데이터
x_plot = np.linspace(x.min(), x.max(), 300).reshape(-1, 1).astype(np.float32)
x_plot_tensor = torch.tensor(x_plot, dtype=torch.float32)

# 선형회귀 모델의 그래프용 예측
with torch.no_grad():
    y_plot_linear = linear_model(x_plot_tensor).numpy()

# 다항회귀 모델은 입력으로 [x, x^2] 형태가 필요함
x_plot_poly = np.column_stack([
    x_plot[:, 0],
    x_plot[:, 0] ** 2
]).astype(np.float32)

x_plot_poly_tensor = torch.tensor(x_plot_poly, dtype=torch.float32)

# 다항회귀 모델의 그래프용 예측
with torch.no_grad():
    y_plot_poly = poly_model(x_plot_poly_tensor).numpy()


# -------------------------------------------------------
# 4. 성능 계산
# -------------------------------------------------------
# R2 계산은 학습 데이터 40개에 대한 예측값으로 계산해야 함
r2_linear = r2_score_np(y, y_pred_linear)
r2_poly = r2_score_np(y, y_pred_poly)

print('r2_linear : ', r2_linear)
print('r2_poly : ', r2_poly)


# -------------------------------------------------------
# 5. 시각화
# -------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.scatter(x, y, label='data')

plt.plot(
    x_plot,
    y_plot_linear,
    label=f'Linear Regression(R2={r2_linear:.3f})'
)

plt.plot(
    x_plot,
    y_plot_poly,
    label=f'Poly Regression(R2={r2_poly:.3f})'
)

plt.xlabel('feature')
plt.ylabel('label')
plt.legend()
plt.grid(True)
plt.show()