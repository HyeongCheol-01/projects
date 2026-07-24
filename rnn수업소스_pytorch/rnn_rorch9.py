실습9)  RNN으로 스팸 메일 분류 모델 (이항 분류)

import re
import copy
import random
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import class_weight

# 실행할 때마다 비슷한 결과가 나오도록 난수 고정
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

data = pd.read_csv('spam.csv',  encoding='latin1')
data = data[['v1', 'v2']]

data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

data.drop_duplicates( subset=['v2'], inplace=True)
data.reset_index( drop=True, inplace=True)

x_data = data['v2']
y_data = data['v1']
print(x_data[:3])
print(y_data[:3])

# 문장을 단어 단위로 변환하는 Tokenizer
class Tokenizer:
    def __init__(self):
        self.word_index = {}

        self.filters = ('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' )

        self.translate_table = str.maketrans({char: ' ' for char in self.filters })

    def tokenize(self, text):
        text = str(text).lower()
        text = text.translate(self.translate_table)

        return text.split()

    def fit_on_texts(self, texts):
        word_counts = Counter()

        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)

        self.word_index = {
            word: index + 1 for index, (word, count) in enumerate(word_counts.most_common())
        }

    def texts_to_sequences(self, texts):
        sequences = []

        for text in texts:
            words = self.tokenize(text)

            sequence = [self.word_index[word] for word in words if word in self.word_index ]
            sequences.append(sequence)

        return sequences


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_data)

sequences = tokenizer.texts_to_sequences(x_data)

word_index = tokenizer.word_index
vocabsize = len(word_index) + 1
print('vocabsize : ', vocabsize)


# 길이가 다른 문장 앞부분에 0을 채워 동일한 길이로 맞춘다.
def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=np.int64 )

    for i, sequence in enumerate(sequences):
        sequence = sequence[-maxlen:]
        if len(sequence) > 0:
            padded[i, -len(sequence):] = sequence

    return padded


# 패딩
max_len = max(len(sequence) for sequence in sequences)
data_pad = pad_sequences( sequences, maxlen=max_len)
print(data_pad[:3])

# train / test split
n_train = int( len(data_pad) * 0.8)

x_train = data_pad[:n_train]
x_test = data_pad[n_train:]

y_train = np.array( y_data[:n_train], dtype=np.float32)
y_test = np.array( y_data[n_train:], dtype=np.float32)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# train 데이터의 마지막 20%를 검증 데이터로 사용
n_val = int( len(x_train) * 0.2)

x_fit = x_train[:-n_val]
y_fit = y_train[:-n_val]

x_val = x_train[-n_val:]
y_val = y_train[-n_val:]


# 데이터 불균형 보정 작업
weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)

class_weights = { 0: weights[0], 1: weights[1]}
print('클래스 가중치 : ', class_weights)


# NumPy 배열을 PyTorch Tensor로 변환
x_fit_tensor = torch.tensor( x_fit, dtype=torch.long)
y_fit_tensor = torch.tensor( y_fit, dtype=torch.float32)
x_val_tensor = torch.tensor( x_val, dtype=torch.long)
y_val_tensor = torch.tensor( y_val, dtype=torch.float32)
x_test_tensor = torch.tensor( x_test, dtype=torch.long)
y_test_tensor = torch.tensor( y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_fit_tensor, y_fit_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset( x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader( test_dataset, batch_size=64, shuffle=False)

# model
class SpamRNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 각 단어를 32차원 실수 벡터로 변환한다.
        self.embedding = nn.Embedding( num_embeddings=vocabsize, embedding_dim=32, 
padding_idx=0)

        self.rnn = nn.RNN(input_size=32, hidden_size=32, batch_first=True )

        # 스팸 여부를 나타내는 하나의 값을 출력한다.
        self.output = nn.Linear( 32, 1 )

    def forward(self, x):
        x = self.embedding(x)
        rnn_output, hidden = self.rnn(x)

        # RNN의 마지막 은닉 상태를 사용한다.
        x = hidden[-1]

        # BCEWithLogitsLoss를 사용하므로
        # Sigmoid를 적용하지 않은 logits를 반환한다.
        logits = self.output(x)

        return logits.squeeze(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용 장치 : ', device)

model = SpamRNNModel().to(device)
print(model)

# 학습할 때 샘플별 손실값을 계산한다.
train_loss_fn = nn.BCEWithLogitsLoss( reduction='none')

# 검증 및 테스트용 손실 함수
eval_loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(model.parameters())

# 클래스 가중치를 PyTorch Tensor로 변환
class_weight_tensor = torch.tensor(
    [
        class_weights[0], class_weights[1]
    ],  dtype=torch.float32, device=device
)

# 검증 또는 테스트 함수
def evaluate(model, data_loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            loss = eval_loss_fn(logits, y_batch)

            total_loss += (loss.item() * len(x_batch) )
            probability = torch.sigmoid(logits)
            predicted = ( probability > 0.5 ).float()
            correct += ( predicted == y_batch  ).sum().item()
            total += len(y_batch)

    return (total_loss / total,  correct / total )


epochs = 300
patience = 3

best_val_loss = float('inf')
best_model_state = None
wait = 0

history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': []
}


for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 이전에 계산된 기울기 초기화
        optimizer.zero_grad( set_to_none=True )

        # 스팸 여부 예측
        logits = model(x_batch)

        # 각 데이터의 이진 교차 엔트로피 손실 계산
        losses = train_loss_fn( logits, y_batch)

        # 실제 클래스에 해당하는 가중치를 가져온다.
        sample_weights = class_weight_tensor[ y_batch.long() ]

        # 클래스 가중치를 적용한 평균 손실 계산
        loss = (losses * sample_weights).mean()
        loss.backward()# 역전파
        optimizer.step()# 가중치 수정

        total_loss += ( loss.item() * len(x_batch) )
        probability = torch.sigmoid(logits)
        predicted = ( probability > 0.5 ).float()
        correct += ( predicted == y_batch ).sum().item()
        total += len(y_batch)

    train_loss = total_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, val_loader )

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(
        f'Epoch {epoch + 1}/{epochs} - '
        f'loss: {train_loss:.4f} - '
        f'acc: {train_acc:.4f} - '
        f'val_loss: {val_loss:.4f} - '
        f'val_acc: {val_acc:.4f}'
    )

    # 검증 손실이 가장 낮으면 모델의 가중치를 저장한다.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy( model.state_dict() )
        wait = 0
    else:
        wait += 1
        # 검증 손실이 3번 연속 개선되지 않으면 학습을 종료한다.
        if wait >= patience:
            print('Early stopping')
            break

model.load_state_dict(best_model_state)  # 검증 손실이 가장 낮았던 모델의 가중치를 복원한다.

# 테스트 데이터 평가
test_loss, test_acc = evaluate( model, test_loader)
print('테스트 정확도 : ', test_acc)

# 성능 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)

plt.plot(history['loss'], label='train loss')
plt.plot(history['val_loss'], label='train val loss')
plt.title('loss over epochs')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot( history['acc'],label='train acc')
plt.plot( history['val_acc'], label='train val acc')
plt.title('acc over epochs')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()

plt.tight_layout()
plt.show()


# 예측
new_mail = [
    "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    "Nah I don't think he goes to usf, he lives around here though",
    "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 1.50 to rcv",
    "Please be at work by 10:00 tomorrow. Have a good rest"
]

new_encoded = tokenizer.texts_to_sequences(new_mail)
new_padded = pad_sequences( new_encoded, maxlen=max_len)

new_tensor = torch.tensor( new_padded, dtype=torch.long, device=device)

model.eval()

with torch.inference_mode():
    logits = model(new_tensor)
    predicted = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)

print(predicted)

for i, mail in enumerate(new_mail):
    prob = predicted[i][0]

    label = ('spam' if prob > 0.5 else 'ham' )
    print(
        f'[{label.upper()}] '
        f'({prob:.4f}) - "{mail}"'
    )
