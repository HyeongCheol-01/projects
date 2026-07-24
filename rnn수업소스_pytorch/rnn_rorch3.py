실습3) LSTM으로 분류 모델 작성(감성분류)
docs = [
    '너무 재밌네요',
    '최고예요',
    '참 잘 만든 영화예요',
    '추천하고 싶은 영화입니다',
    '한 번 더 보고 싶네요',
    '글쎄요',
    '별로네요',
    '생각보다 지루하네요',
    '연기가 어색해요',
    '재미없어요'
]

import numpy as np
import torch
import torch.nn as nn

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
word_index = {}   # 단어 사전 생성

for doc in docs:
    for word in doc.split():
        if word not in word_index:
            word_index[word] = len(word_index) + 1

print(word_index)

# 문장을 단어 번호 시퀀스로 변환
x = []
for doc in docs:
    sequence = [
        word_index[word]
        for word in doc.split()
    ]
    x.append(sequence)

print(x)
# 시퀀스 데이터를 딥러닝 모델에 넣기 전에 토큰의 길이를 동일하게 해야 한다.
maxlen = 5
padded_x = []

for sequence in x:
    sequence = sequence[-maxlen:]
    padding = [0] * (maxlen - len(sequence))
    padded_x.append(padding + sequence)  # 앞부분에 0을 채우는 pre padding

padded_x = np.array(padded_x)
print('패딩결과:\n', padded_x)

# NumPy 배열을 PyTorch Tensor로 변환
x_tensor = torch.tensor(padded_x, dtype=torch.long)
y_tensor = torch.tensor(labels, dtype=torch.float32)
word_size = len(word_index) + 1  # 사용 가능한 토큰 개수 + 1

class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # 각 단어를 8차원 실수 벡터로 표현
        self.embedding = nn.Embedding( num_embeddings=word_size, 
embedding_dim=8, padding_idx=0)
        self.lstm = nn.LSTM(input_size=8,hidden_size=32, batch_first=True)
        self.dense1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]      # LSTM의 마지막 은닉 상태 사용
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x.squeeze(1)

model = SentimentLSTM()
print(model)

# Sigmoid와 Binary Cross Entropy를 함께 처리하는 손실 함수
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 20

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    with torch.no_grad():    # 정확도 계산
        probability = torch.sigmoid(output)
        predicted = ( probability > 0.5).float()
        accuracy = (predicted == y_tensor).float().mean()

    print(
        f'Epoch {epoch + 1:02d} - '
        f'loss: {loss.item():.4f} - '
        f'accuracy: {accuracy.item():.4f}'
    )

model.eval()   # 모델 평가

with torch.inference_mode():
    output = model(x_tensor)
    # output의 값을 0~1 사이 확률값으로 변환(보통 0.5 이상이면 긍정(1), 미만이면 부정(0)으로 분류)
    probability = torch.sigmoid(output)
    predicted = np.where(probability.numpy() > 0.5,1, 0)

    # 예측값과 실제값이 같은 비율, 즉 정확도를 계산
    # 예측값을 1차원으로 펼침 → 텐서로 변환 → 실제값과 비교 → True/False를 1/0으로 변환 → 평균을 구해 정확도 계산
    accuracy = (torch.tensor(predicted.ravel()) == y_tensor).float().mean()
print('정확도 : ', accuracy.item())
print('예측 : ', predicted.ravel())

# BCEWithLogitsLoss가 Sigmoid와 이진 교차 엔트로피를 함께 처리하므로, 모델의 마지막 계층에는 Sigmoid를 넣지 않는다.
