실습6) RNN으로 글자 단위 학습 후 영문 생성


import os
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

filename = 'rnn6text.txt'
with open(filename, encoding='utf-8') as f:
    et = f.read().lower()

print(et[:300] if len(et) > 300 else et)

# 문자 단위 어휘집 생성
chars = sorted(list(set(et)))
print(chars)

char_to_int = {char: index for index, char in enumerate(chars)}
print(char_to_int)

int_to_char = {index: char for index, char in enumerate(chars)}
print(int_to_char)

n_chars = len(et)
n_vocab = len(chars)
print('전체 글자 수 : ', n_chars)
print('전체 어휘 크기 : ', n_vocab)

# 시퀀스 구성
seq_length = 10   # 이전 10개의 글자를 보고 다음 글자 하나를 예측

dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = et[i:i + seq_length]
    seq_out = et[i + seq_length]

    dataX.append([char_to_int[char] for char in seq_in ])
    dataY.append(char_to_int[seq_out])
# print(dataX)
# print(dataY)

N = len(dataX)   # 전체 학습 샘플 개수
print('전체 학습 샘플의 개수 : ', N)

if N == 0:
    raise ValueError('데이터가 적어 학습 시퀀스 생성 불가')


# 입력 데이터 원핫 인코딩
x = F.one_hot( torch.tensor(dataX, dtype=torch.long), num_classes=n_vocab).float()

# CrossEntropyLoss는 정수형태의 정답 번호를 사용하므로 정답 데이터는 원핫 인코딩하지 않는다.
y = torch.tensor( dataY, dtype=torch.long)
print('x shape:', x.shape, ', y shape:', y.shape)

# 학습 데이터 구성
dataset = TensorDataset(x, y)
batch_size = min(8, max(1, N // 2))

train_loader = DataLoader( dataset, batch_size=batch_size, shuffle=True)

# model
class CharLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 입력과 출력 형태: (batch, sequence, feature)
        self.lstm1 = nn.LSTM(input_size=n_vocab, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        self.dense = nn.Linear(128, n_vocab )

    def forward(self, x):
        # 첫 번째 LSTM은 모든 시점의 출력값을 반환한다.
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        # 두 번째 LSTM 처리
        x, _ = self.lstm2(x)

        # 마지막 시점의 출력만 사용한다.
        x = x[:, -1, :]
        x = self.dropout2(x)

        # 다음 글자 후보 전체에 대한 logits를 출력한다.
        return self.dense(x)

model = CharLSTMModel()
print(model)

# CrossEntropyLoss는 Softmax와 다중 분류 손실 계산을 함께 처리한다.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

chkpoint_path = 'data_stru/rnn6model.pt'
os.makedirs( os.path.dirname(chkpoint_path),  exist_ok=True)

epochs = 500
patience = 10

best_loss = float('inf')
best_model_state = None
wait = 0

loss_history = []
accuracy_history = []


# 모델 학습
for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()  # 이전에 계산된 기울기 초기화
        logits = model(x_batch)  # 다음 글자 점수 예측
        loss = criterion(logits, y_batch )# 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 수정
        total_loss += loss.item() * len(x_batch)

        predicted = torch.argmax(logits, dim=1 )
        correct += (predicted == y_batch ).sum().item()
        total += len(y_batch)

    epoch_loss = total_loss / N
    epoch_accuracy = correct / total

    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)

    print(
        f'Epoch {epoch + 1}/{epochs} - '
        f'loss: {epoch_loss:.4f} - '
        f'accuracy: {epoch_accuracy:.4f}'
    )

    # 손실이 가장 낮은 모델 저장
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        wait = 0
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save( best_model_state, chkpoint_path )
    else:
        wait += 1

        # 손실이 patience 횟수 동안 개선되지 않으면 학습 종료
        if wait >= patience:
            print('Early stopping')
            break


# 손실이 가장 낮았던 모델의 가중치 복원
model.load_state_dict(best_model_state)

# 학습 곡선 시각화
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()
loss_ax.plot(loss_history,label='train loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(accuracy_history,label='train accuracy')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='lower left')

plt.tight_layout()
plt.show()


# 샘플링 함수
# 모델이 예측한 확률분포에 temperature와 top_k를 적용해 다음 글자의 인덱스를 무작위로 선택
def sample_with_temperatureFunc(probs, temperature=0.8, top_k=5):
    p = np.asarray( probs,  dtype=np.float64 )
    # 상위 k개의 확률만 남긴다.
    if (top_k is not None and top_k > 0 and top_k < len(p)):
        # 확률 배열 p에서 값이 큰 상위 top_k개의 위치(인덱스)를 찾는 코드
        idx = np.argpartition( p, -top_k)[-top_k:]

        mask = np.zeros_like(p)  # p와 같은 크기의 0 배열을 만든다.
        mask[idx] = p[idx]  # 선택된 상위 k개 위치만 원래 확률을 유지한다.
        p = mask  # 낮은 확률의 후보는 제외한다.

    # temperature로 확률 분포를 조절한다.
    p = np.log(p + 1e-9)
    p = p / max(temperature, 1e-8)
    p = np.exp(p)
    p = p / p.sum()  # 확률의 총합이 1이 되도록 다시 정규화한다.

    # 확률에 따라 다음 글자 번호 하나를 선택한다.
    return int( np.random.choice(len(p), p=p))

# 참고 : np.argpartition(대상 배열, 기준 인덱스)
# 전체를 정렬하지 않고 상위 또는 하위 k개의 인덱스를 찾는다.
# k = 3
# arr = np.array([7, 2, 9, 4, 1])
# idx = np.argpartition(-arr, k - 1)[:k]
# print(idx)

print('문장 생성하기')
start = np.random.randint(0, N)  # 랜덤 시작 인덱스
pattern = list(dataX[start])       # 시작 시퀀스
print(pattern)

seed_text = ''.join(int_to_char[value] for value in pattern)
print(f'seed : "{seed_text}"')

steps = 500       # 생성할 문자 수
temperature = 0.8
top_k = 5
generated = []

model.eval()  # 평가 모드로 변경

# 글자 생성 시에는 기울기를 계산하지 않는다.
with torch.inference_mode():
    for _ in range(steps):
        # 현재 입력 시퀀스를 원핫 인코딩한다.
        x_input = F.one_hot( torch.tensor( [pattern],dtype=torch.long), num_classes=n_vocab).float()

        logits = model(x_input)   # 다음 문자에 대한 logits를 예측한다.

        # logits를 확률값으로 변환한다.
        probs = torch.softmax(logits, dim=1 )[0].numpy()

        idx = sample_with_temperatureFunc(probs, temperature=temperature,top_k=top_k)
        ch = int_to_char[idx]
        generated.append(ch)

        pattern.append(idx)  # 새로 생성한 글자 번호를 입력 시퀀스에 추가한다.
        pattern = pattern[1:]  # 맨 앞의 글자 번호를 제거해 입력 길이를 유지한다.

gen_text = ''.join(generated)
print(gen_text)
