# https://www.kaggle.com/jyotikumarrout/graduation 의
# binary.csv 데이터를 이용하여 미국 대학원 입학여부를 분류하는 모델 작성
# label : admit,  feature : gre, gpa, rank
# PyTorch 이진분류 권장 방식
 # - 출력층에 sigmoid를 직접 넣지 않음
 # - 손실 함수로 BCEWithLogitsLoss 사용
 # - 예측할 때만 torch.sigmoid() 적용

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 랜덤 고정
np.random.seed(42)
torch.manual_seed(42)

# 2. 데이터 읽기
df = pd.read_csv('binary.csv')
print(df.head(3))
print(df.info())

# 3. 전처리 : rank는 연속형 숫자가 아니라 범주형 자료이므로 원핫인코딩 처리
# 예) rank = 1 -> rank_1 = 1, rank_2 = 0, rank_3 = 0, rank_4 = 0
#     rank = 3 -> rank_1 = 0, rank_2 = 0, rank_3 = 1, rank_4 = 0
df = pd.get_dummies(df, columns=['rank'], dtype=int)
print(df.head(3))

# 4. feature, label 분리
x = df.drop('admit', axis=1)
y = df['admit']
print(x.head(3))
print(y.head(3))

# 5. train / test split
# stratify=y:admit 0과 1의 비율이 train/test에 비슷하게 나뉘도록 설정
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y
)

# 6. 스케일링
# 주의: scaler는 train 데이터로 fit, test 데이터는 transform만 해야 함
# 이렇게 해야 test 데이터 정보가 학습 과정에 미리 들어가는 것을 막을 수 있음
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
x_test_scaled = scaler.transform(x_test).astype(np.float32)
y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
y_test_np = y_test.values.reshape(-1, 1).astype(np.float32)

# 7. NumPy 배열을 PyTorch Tensor로 변환
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

print('x_train shape:', x_train_tensor.shape)
print('y_train shape:', y_train_tensor.shape)

# 8. PyTorch 모델 정의
# PyTorch 권장 구조:
# Linear -> ReLU -> Linear -> ReLU -> Linear
# 마지막 sigmoid는 모델 안에 넣지 않음
# BCEWithLogitsLoss가 내부적으로 sigmoid + binary cross entropy를 처리함
class AdmissionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


input_dim = x_train_tensor.shape[1]
model = AdmissionModel(input_dim)
print(model)

# 9. 손실 함수와 optimizer 정의
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10. 평가 함수 정의
def evaluate_model(model, x_data, y_data):
    model.eval()

    with torch.no_grad():
        logits = model(x_data)
        loss = criterion(logits, y_data).item()
        # logits를 sigmoid에 통과시켜 확률값으로 변환
        prob = torch.sigmoid(logits)

        pred = (prob >= 0.5).float()
        acc = (pred == y_data).float().mean().item()

    return loss, acc

# 11. 모델 학습
epochs = 100

history = {
    'loss': [],
    'acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(epochs):
    model.train()  # 학습 모드
    logits = model(x_train_tensor)  # 1) 예측값 계산
    loss = criterion(logits, y_train_tensor)  # 2) 손실 계산
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 학습 정확도 계산
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        train_acc = (pred == y_train_tensor).float().mean().item()

    # 검증 데이터 평가
    val_loss, val_acc = evaluate_model(model, x_test_tensor, y_test_tensor)

    # history 저장
    history['loss'].append(loss.item())
    history['acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(
        f'Epoch {epoch + 1:03d}/{epochs} '
        f'- loss: {loss.item():.4f} '
        f'- acc: {train_acc:.4f} '
        f'- val_loss: {val_loss:.4f} '
        f'- val_acc: {val_acc:.4f}'
    )

# 12. 테스트 데이터 평가
test_loss, test_acc = evaluate_model(model, x_test_tensor, y_test_tensor)
print(f'\n테스트 결과 손실:{test_loss:.4f}, 정확도:{test_acc:.4f}')

# 13. 학습 결과 시각화
plt.figure(figsize=(12, 5))

# loss
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

# accuracy
plt.subplot(1, 2, 2)
plt.plot(history['acc'], label='acc')
plt.plot(history['val_acc'], label='val acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.grid(True)

plt.show()

# 14. 사용자 입력 결과 예측
gre = float(input('gre 점수 입력: '))
gpa = float(input('gpa 학점 입력: '))
rank = int(input('rank 입력(1 ~ 4): '))

# rank 원핫인코딩
rank_encoded = [0, 0, 0, 0]
rank_encoded[rank - 1] = 1

# 입력 순서: gre, gpa, rank_1, rank_2, rank_3, rank_4
user_input = np.array([[gre, gpa] + rank_encoded], dtype=np.float32)
print('user_input : ', user_input)

# 학습 때 사용한 scaler로 동일하게 변환
user_scaled = scaler.transform(user_input).astype(np.float32)
# PyTorch Tensor로 변환
user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
# 예측
model.eval()

with torch.no_grad():
    logits = model(user_tensor)
    # sigmoid를 적용해 합격 확률로 변환
    prob = torch.sigmoid(logits).item()

print('합격 확률 : ', prob)

if prob >= 0.5:
    print('합격 가능성이 높아요')
else:
    print('불합격 가능성이 높아요')