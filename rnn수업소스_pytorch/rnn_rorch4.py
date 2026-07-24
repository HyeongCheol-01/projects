실습4) RNN을 이용한 텍스트 생성
# 문맥을 반영해 다음 단어를 예측하여 텍스트 생성 (다항 분류)

import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# text = """경마장에 있는 말이 뛰고 있다
# 그의 말이 법이다
# 가는 말이 고와야 오는 말이 곱다"""

text = """
미국 행정부가 민간기업 투자를 중심으로 이란 재건에 나설 것으로 알려지면서 국내 산업계도 촉각을 기울이고 있다.
….
국내 정유사들은 종전에 따른 원유 공급망 정상화를 기대하고 있다. 제재 완화 정도에 따라 저렴하고 질 좋은 이란산 원유를 활용할 수 있게 된다면 공급망 다변화 측면에서 선택지를 넓힐 수 있다.
"""

# 단어 단위로 문장을 토큰화
def tokenize(sentence):
    sentence = sentence.lower()

    # 문장부호를 공백으로 변경
    sentence = re.sub(
        r'[!"#$%&()*+,\-./:;<=>?@\[\]\\^_`{|}~\t]',
        ' ',
        sentence
    )
    return sentence.split()


# 단어 사전 생성
words = tokenize(text)
word_counts = Counter(words)
# 빈도수가 높은 단어부터 번호를 부여
word_index = {
    word: index + 1
    for index, (word, count) in enumerate(
        word_counts.most_common()
    )
}
print(word_index)


# 문장을 단어 번호 시퀀스로 변환
def texts_to_sequences(sentence):
    return [ word_index[word] for word in tokenize(sentence) if word in word_index ]

encoded = texts_to_sequences(text)
print(encoded)

vocab_size = len(word_index) + 1  # 실제 단어 집합 + 패딩 번호 0

# 훈련 데이터 작성
sequences = list()

for line in text.split('\n'):    # 줄 단위로 문장 분리
    enco = texts_to_sequences(line)
    # 바로 다음 단어를 label로 사용하기 위해 리스트에 저장
    for i in range(1, len(enco)):
        sequ = enco[:i + 1]
        sequences.append(sequ)

print('학습에 참여할 샘플 수 : ', len(sequences))
print(sequences)
print(max(len(i) for i in sequences))

# 전체 시퀀스의 길이를 동일하게 맞추는 함수
def pad_sequences(sequences, maxlen):
    result = []

    for sequence in sequences:
        sequence = sequence[-maxlen:]

        # 앞부분에 0을 채우는 pre padding
        padding = [0] * (maxlen - len(sequence))
        result.append(padding + sequence)

    return np.array(result, dtype=np.int64)

max_len = max(len(i) for i in sequences)
psequences = pad_sequences(sequences, maxlen=max_len)
print(psequences)

print()
# 각 시퀀스의 마지막 요소를 label로 사용하기 위해 분리
x = psequences[:, :-1]    # feature
y = psequences[:, -1]     # label
print(x)
print(y)

# CrossEntropyLoss는 정수 형태의 label을 사용하므로
# label을 원-핫 인코딩하지 않는다.
x_tensor = torch.tensor(x, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32,shuffle=True)

# model
class TextGenerationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 각 단어를 32차원 실수 벡터로 표현
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
embedding_dim=32,padding_idx=0)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)

        self.dense1 = nn.Linear(32, 32)
        self.relu1 = nn.ReLU()

        self.dense2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()

        self.output = nn.Linear(16, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        lstm_output, (hidden, cell) = self.lstm(x)

        # 마지막 시점의 LSTM 출력 사용
        x = lstm_output[:, -1, :]

        x = self.dense1(x)
        x = self.relu1(x)

        x = self.dense2(x)
        x = self.relu2(x)

        # CrossEntropyLoss를 사용하므로 Softmax를 적용하지 않는다.
        x = self.output(x)

        return x


model = TextGenerationModel()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 200

for epoch in range(epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        output = model(x_batch)
        loss = criterion(output,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x_batch)

        predicted = torch.argmax(output, dim=1)
        correct += (predicted == y_batch).sum().item()
        total += len(y_batch)

    epoch_loss = total_loss / len(dataset)
    epoch_accuracy = correct / total

    print(
        f'Epoch {epoch + 1}/{epochs} - '
        f'loss: {epoch_loss:.4f} - '
        f'accuracy: {epoch_accuracy:.4f}'
    )


# 모델 평가
model.eval()

with torch.inference_mode():
    output = model(x_tensor)
    loss = criterion(output, y_tensor)
    predicted = torch.argmax(output, dim=1)
    accuracy = (predicted == y_tensor).float().mean()

print('평가 결과 : ', loss.item(), accuracy.item())

# 단어 번호를 단어로 변환하기 위한 사전
index_word = {index: word for word, index in word_index.items()}

# 문자열 생성 함수
def sequence_gen_text(model, current_word, n):
    init_word = current_word
    sentence = ''

    model.eval()

    with torch.inference_mode():
        for _ in range(n):
            encoded = texts_to_sequences(current_word)
            encoded = pad_sequences([encoded], maxlen=max_len - 1)
            encoded_tensor = torch.tensor(encoded, dtype=torch.long)

            output = model(encoded_tensor)

            # 가장 높은 값을 가진 단어 번호 선택
            result = torch.argmax(output,dim=1).item()

            # 예측 단어 찾기
            word = index_word[result]

            current_word = current_word + ' ' + word
            sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence


# print(sequence_gen_text(model, '경마장', 5))
# print(sequence_gen_text(model, '그의', 5))
# print(sequence_gen_text(model, '고와야', 5))

print(sequence_gen_text(model, '미국 행정부', 20))
print(sequence_gen_text(model, '제재', 20))
print(sequence_gen_text(model, '세계적인', 20))
