# 다항분류는 여러 개의 class 중 하나를 분류하는 문제
# PyTorch에서는 보통 출력층에 softmax를 직접 넣지 않고
# nn.CrossEntropyLoss()를 사용함
# 이유: nn.CrossEntropyLoss()가 내부적으로
# LogSoftmax + Negative Log Likelihood Loss를 함께 처리하기 때문

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1. softmax 함수 직접 확인
# softmax 함수: 입력받은 실수 벡터를 0~1 사이의 확률값으로 변환
# 모든 출력값의 합은 1이 됨
# 수식: softmax(x_i) = exp(x_i) / sum(exp(x))
# c = np.max(a)를 빼는 이유: exp 계산 시 너무 큰 값이 나와 overflow 되는 것을 방지하기 위함
def softmaxFunc(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

data = np.array([0.3, 2.8, 4.0])
print(softmaxFunc(data))


# 2. 예제 데이터 생성
np.random.seed(1)
torch.manual_seed(1)
np.set_printoptions(suppress=True, precision=3)

# xdata : feature - 예시로 12개 시험 점수/특징이 있다고 가정
xdata = np.random.random((1000, 12)).astype(np.float32)
# ydata : label - 0, 1, 2, 3, 4 중 하나. 다섯 과목 중 하나로 분류한다고 가정
ydata = np.random.randint(5, size=(1000,))
print(xdata[:2])
print(ydata[:2])

# 3. PyTorch Tensor로 변환
x_tensor = torch.tensor(xdata, dtype=torch.float32)
y_tensor = torch.tensor(ydata, dtype=torch.long)
print("x_tensor shape:", x_tensor.shape)  # torch.Size([1000, 12])
print("y_tensor shape:", y_tensor.shape)  # torch.Size([1000])

# 4. PyTorch 다항분류 모델 정의
class MultiClassClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 5)
        )

    def forward(self, x):
        return self.net(x)

model = MultiClassClassifier()
print(model)

# 5. 손실 함수와 optimizer 정의
# CrossEntropyLoss: 다항분류용 손실 함수, 모델 출력값 logits와 정수 label을 받음
# optimizer: Adam 사용
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 모델 학습
epochs = 2000
batch_size = 32

history = {
    "loss": [],
    "accuracy": []
}

data_size = len(x_tensor)

for epoch in range(epochs):
    model.train()

    # 매 epoch마다 데이터 섞기
    indices = torch.randperm(data_size)

    epoch_loss_sum = 0
    epoch_correct = 0
    epoch_total = 0

    for start in range(0, data_size, batch_size):
        batch_idx = indices[start:start + batch_size]

        x_batch = x_tensor[batch_idx]
        y_batch = y_tensor[batch_idx]

        # logits shape: (batch_size, 5)
        logits = model(x_batch)
        # CrossEntropyLoss는 softmax 전의 logits를 입력으로 받음
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        # 5) 파라미터 업데이트
        optimizer.step()
        # loss 누적
        epoch_loss_sum += loss.item() * x_batch.size(0)

        # 예측 class 계산
        pred_class = torch.argmax(logits, dim=1)

        epoch_correct += (pred_class == y_batch).sum().item()
        epoch_total += y_batch.size(0)

    epoch_loss = epoch_loss_sum / epoch_total
    epoch_acc = epoch_correct / epoch_total

    history["loss"].append(epoch_loss)
    history["accuracy"].append(epoch_acc)

    print(
        f"Epoch {epoch + 1:04d}/{epochs} "
        f"- loss: {epoch_loss:.4f} "
        f"- accuracy: {epoch_acc:.4f}"
    )

# 7. 모델 평가
model.eval()

with torch.no_grad():
    logits = model(x_tensor)

    loss = criterion(logits, y_tensor).item()

    pred_class = torch.argmax(logits, dim=1)
    acc = (pred_class == y_tensor).float().mean().item()

print("모델 평가 결과 : ", [loss, acc])

# 8. 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["loss"])
ax1.set_title("Loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.grid(True)

ax2.plot(history["accuracy"])
ax2.set_title("Accuracy")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.grid(True)
plt.show()

# 9. 기존 값으로 분류 예측
model.eval()

with torch.no_grad():
    sample_x = x_tensor[:5]

    # logits: softmax 적용 전 점수
    sample_logits = model(sample_x)

    # softmax를 적용해 class별 확률로 변환
    sample_prob = torch.softmax(sample_logits, dim=1)

    # 가장 큰 확률의 index가 예측 class
    sample_pred_class = torch.argmax(sample_prob, dim=1)

print("예측 확률 : ", sample_prob.numpy())
print("예측값 : ", sample_pred_class.numpy())
print("실제값 : ", y_tensor[:5].numpy())

# 10. 새로운 값으로 예측
x_new = np.random.random([1, 12]).astype(np.float32)
print(x_new)
x_new_tensor = torch.tensor(x_new, dtype=torch.float32)

with torch.no_grad():
    new_logits = model(x_new_tensor)
    # class별 확률
    new_pred = torch.softmax(new_logits, dim=1)
    # 가장 큰 확률의 class index
    new_pred_class = torch.argmax(new_pred, dim=1)

print("분류 결과 : ", new_pred.numpy())
print("분류 결과합 : ", torch.sum(new_pred).item())
print("분류 결과 : ", new_pred_class.item())

# 예측 결과를 과목명으로 출력하기
classes = np.array(["국어", "영어", "수학", "과학", "체육"])
print("예측값 : ", classes[new_pred_class.item()])