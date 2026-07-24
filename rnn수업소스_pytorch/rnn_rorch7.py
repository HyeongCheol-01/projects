실습7)  토지 소설을 글자 단위로 학습한 후 소설쓰기

import os
import copy
import random
import re
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 데이터 파일 다운로드
url = 'https://raw.githubusercontent.com/pykwon/etc/master/rnn_test_toji.txt'
path = Path('rnn_test_toji.txt')

if not path.exists():
    urlretrieve(url, path)

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

print('글자 수 : ', len(text))              # 전체 글자 수를 출력
print('행 수 : ', len(text.splitlines()))   # 전체 행 수를 출력
print(text[:300])

# 텍스트 전처리
text = re.sub('[^가-힣 .,?!]', '', text)  # 한글, 공백, 기본 구두점만 남긴다
text = re.sub(' +', ' ', text)            # 연속된 공백을 하나의 공백으로 줄인다
text = text.strip()
print('전처리 후 글자 수 : ', len(text))
print('전처리 후 행 수 : ', len(text.splitlines()))

# 고유 문자 정의
chars = sorted(list(set(text)))    # 텍스트에 등장한 고유 문자 목록을 만든다
vocab_size = len(chars)          # 고유 문자 개수를 저장한다
print('사용 가능 문자 수 : ', vocab_size)

# 문자와 인덱스 매핑
char_indices = { char: index for index, char in enumerate(chars)}  # 문자 → 숫자
indices_char = {index: char for index, char in enumerate(chars)}  # 숫자 → 문자


# 학습 데이터 준비
maxlen = 30  # 입력 시퀀스 길이
step = 10    # 시퀀스를 자르는 간격을 정한다

sentences = []   # 입력 시퀀스를 저장할 리스트
next_chars = []  # 정답 글자를 저장할 리스트

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])  # 30글자 입력 문장을 저장
    next_chars.append(text[i + maxlen])   # 그 다음 글자를 정답으로 저장한다
print('시퀀스 개수 : ', len(sentences))

if len(sentences) == 0:
    raise ValueError('데이터가 적어 학습 시퀀스를 만들 수 없다.')


# 정수 인코딩
x = np.zeros( (len(sentences), maxlen), dtype=np.int64)  # 입력 문장을 정수 배열로 저장한다
y = np.zeros( (len(sentences),), dtype=np.int64)          # 정답 글자를 정수 배열로 저장한다

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t] = char_indices[char]  # 글자를 숫자 인덱스로 변환해 저장

    y[i] = char_indices[next_chars[i]]  # 정답 글자도 숫자 인덱스로 저장
print('x shape : ', x.shape)
print('y shape : ', y.shape)

# NumPy 배열을 PyTorch Tensor로 변환
x_tensor = torch.tensor(x, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# 학습 데이터 구성
dataset = TensorDataset( x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 모델 구성
class CharLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 정수 인덱스를 64차원 벡터로 변환한다.
        self.embedding = nn.Embedding(num_embeddings=vocab_size,  embedding_dim=64 )

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        # 다음 문자 후보 전체에 대한 점수를 출력한다.
        self.output = nn.Linear(128, vocab_size )

    def forward(self, x):
        x = self.embedding(x)

        # LSTM 출력 형태: (배치 크기, 시퀀스 길이, 은닉 상태 크기)
        x, (hidden, cell) = self.lstm(x)

        # 마지막 시점의 출력만 사용한다.
        x = x[:, -1, :]
        x = self.dropout(x)

        # CrossEntropyLoss를 사용하므로 Softmax는 적용하지 않는다.
        logits = self.output(x)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용 장치 : ', device)

model = CharLSTMModel().to(device)
print(model)

# 손실 함수와 최적화 알고리즘
# CrossEntropyLoss는 정수 형태의 정답 인덱스를 사용한다.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

# 모델 학습
checkpoint_path = 'best_model.pt'

epochs = 10
patience = 3

best_loss = float('inf')
best_model_state = None
wait = 0

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 이전 배치에서 계산된 기울기를 초기화한다.
        optimizer.zero_grad(set_to_none=True)

        logits = model(x_batch)  # 다음 글자에 대한 점수를 예측한다.
        loss = criterion(logits, y_batch )  # 손실을 계산한다.
        loss.backward()# 역전파를 수행한다.
        optimizer.step()# 모델의 가중치를 수정한다.

        total_loss += loss.item() * len(x_batch)
        total_count += len(x_batch)

    epoch_loss = total_loss / total_count
    print(
        f'Epoch {epoch + 1}/{epochs} - '
        f'loss: {epoch_loss:.4f}'
    )

    # 현재 손실이 가장 낮으면 모델 가중치를 저장한다.
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        wait = 0
        best_model_state = copy.deepcopy( model.state_dict())
        torch.save(best_model_state, checkpoint_path )
    else:
        wait += 1
        # 손실이 patience 횟수 동안 개선되지 않으면 학습을 종료한다.
        if wait >= patience:
            print('Early stopping')
            break

# 손실이 가장 낮았던 모델의 가중치를 복원한다.
model.load_state_dict(best_model_state)

# 모델 저장
torch.save(model.state_dict(), 'char_rnn_model.pt')

# 샘플링 함수 정의
def sample(preds, temperature=0.5):
    preds = np.asarray(preds, dtype=np.float64 )   # 예측 확률을 실수 배열로 변환한다.

    # temperature로 확률 분포를 조절한다.
    preds = np.log(preds + 1e-8) / max(temperature, 1e-8 )
    exp_preds = np.exp(preds - np.max(preds))  # 지수 함수를 적용한다.
    preds = exp_preds / np.sum(exp_preds)  # 전체 합이 1이 되도록 정규화한다.
    # 확률 분포에 따라 문자 하나를 선택한다.
    probas = np.random.multinomial(1, preds, 1 )
    return np.argmax(probas)  # 선택된 문자의 인덱스를 반환한다.

# 시작 문장 준비
start_index = random.randint(0, len(text) - maxlen - 1)

# 시작 위치에서 30글자를 가져온다.
seed_text = text[ start_index:start_index + maxlen]

# 예측에 사용할 최근 30글자를 저장한다.
generated_text = seed_text

# 최종 결과를 저장한다.
final_text = seed_text
print('시작 문장 : ', seed_text)
print('\n생성 시작...\n')

# 텍스트 생성
model.eval()

with torch.inference_mode():
    for i in range(1000):  # 1000글자를 생성한다.
        sampled = np.zeros((1, maxlen), dtype=np.int64)

        # 최근 30글자를 정수 인덱스로 변환한다.
        for t, char in enumerate(generated_text):
            sampled[0, t] = char_indices[char]

        sampled_tensor = torch.tensor( sampled, dtype=torch.long, device=device)
        logits = model(sampled_tensor)       # 다음 글자에 대한 logits를 예측한다.
        preds = torch.softmax(logits, dim=1 )[0].cpu().numpy()  # logits를 확률로 변환한다.

        # 예측 확률에서 다음 글자를 선택한다.
        next_index = sample( preds, temperature=0.5)

        next_char = indices_char[next_index]   # 인덱스를 문자로 변환한다.
        generated_text += next_char         # 예측한 글자를 뒤에 붙인다.
        generated_text = generated_text[1:]   # 맨 앞 글자를 제거해 길이를 30으로 유지한다.
        final_text += next_char               # 최종 결과에 예측 글자를 추가한다.
        print(next_char, end='',  flush=True )  # 생성된 글자를 바로 출력한다.

print('\n\n생성된 텍스트:\n')
print(final_text)

# 텍스트 저장
with open( 'generated_text.txt','w', encoding='utf-8') as f:
    f.write(final_text)

print('\n텍스트 저장 완료 → generated_text.txt')
