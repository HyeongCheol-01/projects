# 다항회귀
# 광고비와 매출의 관계가 직선이 아니라 곡선 형태인 데이터를 대상으로 함
# 선형회귀:  매출 = 광고비 * w + b
# 다항회귀:  매출 = 광고비 * w1 + 광고비^2 * w2 + b
# 핵심: 모델 자체는 nn.Linear를 사용하지만,
#   입력 feature를 [x, x^2] 형태로 확장하여 곡선 관계를 학습한다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import torch
import torch.nn as nn
import torch.optim as optim

# 랜덤 고정
np.random.seed(7)
torch.manual_seed(7)

# 1. 다항회귀에 적합한 예제 데이터 생성
# 광고비가 증가하면 매출도 증가하지만, 어느 정도 이후에는 증가폭이 둔화되는 곡선 데이터
ad_cost = np.linspace(0, 100, 80)

# sales = -0.06 * 광고비^2 + 7.5 * 광고비 + 40 + noise
# 실제로는 2차 함수 형태의 관계를 갖도록 인위적으로 생성
sales = (
    -0.06 * (ad_cost ** 2)
    + 7.5 * ad_cost
    + 40
    + np.random.normal(0, 25, size=len(ad_cost))
)

df = pd.DataFrame({
    '광고비': ad_cost,
    '매출': sales
})

print(df.head())

# CSV 파일로 저장
df.to_csv('ad_sales.csv', index=False, encoding='utf-8-sig')
print('csv 저장 성공')

# CSV 파일 읽기
df = pd.read_csv('ad_sales.csv')
print(df.info())

# 결측치가 있다면 해당 행 삭제
df = df.dropna()
print('데이터 크기 : ', df.shape)

# 2. feature, label 분리
# x : 입력 데이터, 광고비
# y : 정답 데이터, 매출
x = df[['광고비']].values.astype(np.float32)
y = df[['매출']].values.astype(np.float32)

print(x[:3])
print(y[:3])

# 산점도 확인
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.7)
plt.xlabel('광고비')
plt.ylabel('매출액')
plt.grid(True)
plt.show()

# 3. train / test split
# sklearn 없이 numpy로 직접 데이터 섞기
indices = np.arange(len(x))
np.random.shuffle(indices)

x = x[indices]
y = y[indices]

train_size = int(len(x) * 0.8)

x_train = x[:train_size]
x_test = x[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print('x:', x_train.shape, x_test.shape)  # (64, 1) (16, 1)
print('y:', y_train.shape, y_test.shape)

# 4. Scaling
# train 데이터 기준으로 평균과 표준편차 계산
# test 데이터도 반드시 train 기준 평균/표준편차로 표준화해야 함
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)

# 입력값 표준화
x_train_scaled = (x_train - x_mean) / x_std
x_test_scaled = (x_test - x_mean) / x_std

# y도 표준화
# PyTorch 학습 시 손실 크기를 안정적으로 만들기 위함
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)

y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# 5. 다항 특성 생성 함수
# degree=2이면 [x, x^2] 생성
# degree=3이면 [x, x^2, x^3] 생성
#
# 주의:
#   다항 특성은 스케일링된 x를 기준으로 만드는 것이 안정적임
def make_poly_features(x_scaled, degree=2):
    features = [x_scaled ** d for d in range(1, degree + 1)]
    return np.concatenate(features, axis=1).astype(np.float32)

# 선형회귀용 입력: [x]
x_train_linear = x_train_scaled
x_test_linear = x_test_scaled

# 다항회귀용 입력: [x, x^2]
x_train_poly = make_poly_features(x_train_scaled, degree=2)
x_test_poly = make_poly_features(x_test_scaled, degree=2)

print('선형회귀 입력 shape : ', x_train_linear.shape)
print('다항회귀 입력 shape : ', x_train_poly.shape)
print(x_train_linear[:2])
print(x_train_poly[:2])

# 6. NumPy 데이터를 PyTorch Tensor로 변환
# PyTorch 모델은 Tensor를 입력으로 받음
x_train_linear_tensor = torch.tensor(x_train_linear, dtype=torch.float32)
x_test_linear_tensor = torch.tensor(x_test_linear, dtype=torch.float32)

x_train_poly_tensor = torch.tensor(x_train_poly, dtype=torch.float32)
x_test_poly_tensor = torch.tensor(x_test_poly, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# 7. R2 score 계산 함수
def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)          # 잔차 제곱합
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # 전체 제곱합
    return 1 - (ss_res / ss_tot)

# 모델 성능 평가 함수
def evaluate_model(name, y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score_np(y_true, y_pred)

    print(f'\n[{name}]')
    print('MSE : ', round(mse, 3))
    print('RMSE : ', round(rmse, 3))
    print('R2 : ', round(r2, 3))

# 8. PyTorch 학습 함수 정의
# TensorFlow의 model.fit() 대신
# PyTorch에서는 학습 과정을 직접 작성한다.
def train_model(model, x_train_tensor, y_train_tensor, epochs=2000, lr=0.01):
    # 손실 함수 : 평균제곱오차
    criterion = nn.MSELoss()

    # 옵티마이저 : Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # 1) 모델 예측값 계산
        y_pred = model(x_train_tensor)

        # 2) 손실 계산
        loss = criterion(y_pred, y_train_tensor)

        # 3) 이전 epoch에서 계산된 gradient 초기화
        optimizer.zero_grad()

        # 4) 역전파로 gradient 계산
        loss.backward()

        # 5) 파라미터 업데이트
        optimizer.step()

    return model

# 9. 선형회귀 모델 학습
# 입력 feature 1개 → 출력 1개
# TensorFlow Dense(units=1)과 유사한 역할
linear_model = nn.Linear(in_features=1, out_features=1)

linear_model = train_model(
    model=linear_model,
    x_train_tensor=x_train_linear_tensor,
    y_train_tensor=y_train_tensor,
    epochs=2000,
    lr=0.01
)

# 평가 시에는 gradient 계산이 필요 없으므로 torch.no_grad() 사용
with torch.no_grad():
    y_pred_linear_scaled = linear_model(x_test_linear_tensor).numpy()

# 표준화된 예측값을 원래 매출 단위로 복원
y_pred_linear = y_pred_linear_scaled * y_std + y_mean

# 10. 다항회귀 모델 학습
# 입력 feature 2개([x, x^2]) → 출력 1개
# 모델은 선형 계층 하나지만,
# 입력에 x^2이 포함되어 있으므로 곡선 형태를 학습할 수 있음
poly_model = nn.Linear(in_features=2, out_features=1)

poly_model = train_model(
    model=poly_model,
    x_train_tensor=x_train_poly_tensor,
    y_train_tensor=y_train_tensor,
    epochs=2000,
    lr=0.01
)

with torch.no_grad():
    y_pred_poly_scaled = poly_model(x_test_poly_tensor).numpy()

# 표준화된 예측값을 원래 매출 단위로 복원
y_pred_poly = y_pred_poly_scaled * y_std + y_mean


# 11. 성능 비교
evaluate_model('선형회귀', y_test, y_pred_linear)
evaluate_model('다항회귀(degree=2)', y_test, y_pred_poly)

# 12. 시각화
# 부드러운 회귀선을 그리기 위한 x축 데이터 생성
x_plot = np.linspace(x.min(), x.max(), 300).reshape(-1, 1).astype(np.float32)

# 그래프용 x도 train 기준 평균/표준편차로 표준화
x_plot_scaled = (x_plot - x_mean) / x_std

# 다항회귀용 그래프 입력 생성
x_plot_poly = make_poly_features(x_plot_scaled, degree=2)

# PyTorch Tensor 변환
x_plot_scaled_tensor = torch.tensor(x_plot_scaled, dtype=torch.float32)
x_plot_poly_tensor = torch.tensor(x_plot_poly, dtype=torch.float32)

# 그래프용 예측
with torch.no_grad():
    y_plot_linear_scaled = linear_model(x_plot_scaled_tensor).numpy()
    y_plot_poly_scaled = poly_model(x_plot_poly_tensor).numpy()

# 예측값을 원래 매출 단위로 복원
y_plot_linear = y_plot_linear_scaled * y_std + y_mean
y_plot_poly = y_plot_poly_scaled * y_std + y_mean

plt.figure(figsize=(9, 6))
plt.scatter(x_train, y_train, alpha=0.5, label='train data')
plt.scatter(x_test, y_test, alpha=0.9, label='test data')
plt.plot(x_plot, y_plot_linear, label='선형회귀')
plt.plot(x_plot, y_plot_poly, label='다항회귀(degree=2)')
plt.xlabel('광고비')
plt.ylabel('매출')
plt.legend()
plt.grid(True)
plt.show()