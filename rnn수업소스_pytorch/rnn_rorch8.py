실습8) 자모 기반 언어 모델

# 주피터 노트북에서 최초 한 번 실행
!pip install jamotools

# 자모 분리 테스트
import sys
from pathlib import Path
from urllib.request import urlretrieve
import jamotools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


url = 'https://raw.githubusercontent.com/pykwon/etc/master/rnn_test_toji.txt'
path_to_file = Path('toji.txt')

# 파일이 없으면 다운로드
if not path_to_file.exists():
    urlretrieve(url, path_to_file)

# open(path_to_file, 'rb'): rb는 "read binary", 즉 바이트 단위로 읽기를 의미
# 바이트 단위로 읽으면 bytes 객체가 반환된다.
# decode('utf-8'): bytes를 UTF-8 문자열로 변환한다.
train_text = open( path_to_file, 'rb').read().decode(encoding='utf-8')
s = train_text[:100]
print('s:', s)

# 한글 텍스트를 자모 단위로 분리: 숫자, 기호, 영어, 한자 등에는 영향을 주지 않는다.
s_split = jamotools.split_syllables(s)
print('s_split:', s_split)

# 자모 결합 테스트
s2 = jamotools.join_jamos(s_split)
print('s2:', s2)
print(s == s2)

# 자모 토큰화
train_text_X = jamotools.split_syllables(train_text)

# 자모 문자열 전체에서 중복 없는 고유 문자 집합을 만들고 정렬한다.
vocab = sorted(set(train_text_X))

# 모델이 학습 중 본 적 없는 문자를 처리하기 위한 특수 토큰
vocab.append('UNK')
print('{} unique characters'.format(len(vocab)))

# 자모 문자를 정수로 변환하기 위한 사전
char2idx = { char: index for index, char in enumerate(vocab)}

# 각 자모를 정수로 변환해 NumPy 배열로 만든다.
text_as_int = np.array(
    [
        char2idx[char]
        for char in train_text_X
    ], dtype=np.int64
)
print(text_as_int)
print('index of UNK: {}'.format(char2idx['UNK']))

# 토큰 데이터 확인
print(train_text_X[:20])
print(text_as_int[:20])


# 학습 데이터세트 생성 : 모델은 80개의 자모를 입력받아 다음 자모 하나를 예측한다.
seq_length = 80
# 숫자 인덱스를 다시 자모 문자로 변환하기 위한 배열
idx2char = np.array( vocab, dtype=object)

# 자모 인덱스를 seq_length + 1개씩 묶는다.앞의 80개는 입력, 마지막 1개는 정답으로 사용한다.
chunk_size = seq_length + 1

num_chunks = len(text_as_int) // chunk_size

if num_chunks == 0:
    raise ValueError('텍스트가 너무 짧아 학습 데이터를 만들 수 없다.')

# chunk_size로 나누어 떨어지지 않는 마지막 데이터는 제외한다.
usable_length = num_chunks * chunk_size

chunks = torch.tensor(text_as_int[:usable_length], dtype=torch.long).reshape(
    num_chunks, chunk_size
)

# 데이터 확인
item = chunks[0]
print(idx2char[item.numpy()])
print(item.numpy())

# 입력 시퀀스와 정답 분리
# chunk[:-1]: 앞 80개 자모
# chunk[-1]: 다음에 올 정답 자모
x = chunks[:, :-1]
y = chunks[:, -1]

# 입력과 정답 확인
print(idx2char[x[0].numpy()])
print(x[0].numpy())
print(idx2char[y[0].item()])
print(y[0].item())

BATCH_SIZE = 64

# 한 번의 학습에서 입력 80개와 정답 1개로 구성된 데이터 샘플을 최대 64개씩 처리한다.
train_dataset = TensorDataset(x, y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 한 에포크에서 처리할 미니 배치 수
steps_per_epoch = len(train_loader)
print('steps_per_epoch:', steps_per_epoch)

# 자모 단위 생성 모델 정의
total_chars = len(vocab)

class JamoLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 자모 인덱스를 100차원 벡터로 변환한다.
        self.embedding = nn.Embedding(num_embeddings=total_chars, embedding_dim=100)

        # 400개의 은닉 상태를 가진 LSTM
        self.lstm = nn.LSTM(input_size=100, hidden_size=400, batch_first=True)

        # 전체 자모 문자에 대한 예측 점수를 출력한다.
        self.dense = nn.Linear(400, total_chars )

    def forward(self, x):
        # 입력 형태: (batch_size, sequence_length)
        x = self.embedding(x)

        # 임베딩 결과: (batch_size, sequence_length, 100)
        output, (hidden, cell) = self.lstm(x)

        # 마지막 시점의 출력만 사용한다.
        output = output[:, -1, :]

        # CrossEntropyLoss를 사용하므로 Softmax를 적용하지 않은 logits를 반환한다.
        logits = self.dense(output)

        return logits


# GPU가 있으면 GPU를 사용하고 없으면 CPU를 사용한다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용 장치:', device)

model = JamoLSTMModel().to(device)
print(model)

# 정수 형태의 정답 자모 인덱스를 사용하는 다중 분류 손실 함수
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 모델의 logits로부터 다음 자모 인덱스를 선택한다.
# 단순히 가장 큰 값을 선택하지 않고 확률에 따라 무작위로 선택한다.
def sample(logits, temperature=0.7):
    # temperature가 0이 되는 것을 방지한다.
    temperature = max(temperature, 1e-8 )

    # temperature가 낮으면 높은 점수의 문자가 선택될 가능성이 커지고,
    # 높으면 다양한 문자가 선택될 가능성이 커진다.
    logits = logits / temperature
    probs = torch.softmax(logits, dim=0)   # logits를 확률 분포로 변환한다.

    # 확률 분포에 따라 자모 인덱스 하나를 선택한다.
    return torch.multinomial( probs, num_samples=1 ).item()


# 입력 자모 시퀀스의 길이를 seq_length로 맞추는 함수
def make_input_sequence(test_sentence):
    test_text_X = test_sentence[-seq_length:]   # 최근 80개의 자모만 사용한다.

    # 사전에 없는 문자는 UNK로 변환한다.
    test_ids = [char2idx.get( char, char2idx['UNK']) for char in test_text_X]
    # 길이가 부족하면 앞쪽을 UNK로 채운다.
    padding_size = seq_length - len(test_ids)
    test_ids = ([char2idx['UNK']] * padding_size + test_ids)
    return torch.tensor([test_ids], dtype=torch.long, device=device )


# 학습된 모델을 이용해 자모 문자열을 생성하는 함수
def generate_jamos( seed_text, next_chars=300, temperature=0.7, stream_output=True):
    # 시작 문장을 자모 단위로 분리한다.
    test_sentence = jamotools.split_syllables( seed_text)

    model.eval()

    with torch.inference_mode():
        for _ in range(next_chars):
            # 최근 80개 자모를 모델 입력으로 만든다.
            test_text_X = make_input_sequence( test_sentence )

            # 다음 자모에 대한 logits를 예측한다.
            output_logits = model(test_text_X)[0]

            # 확률에 따라 다음 자모 하나를 선택한다.
            output_idx = sample(output_logits, temperature=temperature)

            output_char = idx2char[output_idx]

            # 예측한 자모를 기존 문장에 추가한다.
            test_sentence += output_char

            if stream_output:
                sys.stdout.write(output_char)
                sys.stdout.flush()

    return test_sentence


# 모델 학습 중간에 생성 결과를 출력하는 함수
def testmodel2(current_epoch, total_epochs):
    # 5에포크마다 또는 마지막 에포크에 실행한다.
    if (current_epoch % 5 != 0 and current_epoch != total_epochs):
        return

    # 원본 텍스트의 처음 48글자를 시작 문장으로 사용한다.
    seed_text = train_text[:48]

    generated_jamos = generate_jamos( seed_text, next_chars=300, temperature=0.7, 
stream_output=True )
    print('\n')
    # 자모를 한글 음절로 결합한다.
    print('\nGenerated sentence:\n')
    print(jamotools.join_jamos(generated_jamos))


EPOCHS = 5
loss_history = []
accuracy_history = []

# 모델 학습
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct_count = 0
    total_count = 0

    for x_batch, y_batch in train_loader:
        # 입력과 정답을 CPU 또는 GPU로 이동한다.
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 이전에 계산된 기울기를 초기화한다.
        optimizer.zero_grad(set_to_none=True)

        # 다음 자모의 logits를 예측한다.
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch )   # 손실을 계산한다.
        loss.backward()       # 역전파를 실행한다.
        optimizer.step()      # 모델의 가중치를 수정한다.

        total_loss += loss.item() * len(x_batch)
        predicted = torch.argmax(logits, dim=1 )

        correct_count += (predicted == y_batch).sum().item()
        total_count += len(y_batch)

    epoch_loss = total_loss / total_count
    epoch_accuracy = correct_count / total_count

    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)

    print(
        f'Epoch {epoch + 1}/{EPOCHS} - '
        f'loss: {epoch_loss:.4f} - '
        f'accuracy: {epoch_accuracy:.4f}'
    )

    # 에포크가 끝날 때 생성 결과를 확인한다.
    testmodel2( current_epoch=epoch + 1, total_epochs=EPOCHS)


# 모델과 자모 사전 저장
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'seq_length': seq_length,
        'embedding_dim': 100,
        'hidden_size': 400
    },
    'rnnmodel.pt'
)


# 임의의 문장을 사용한 생성 결과 확인
test_sentence = '최참판댁 사랑은 무인지경처럼 적막하다'
generated_jamos = generate_jamos(test_sentence, next_chars=500, temperature=0.7, 
stream_output=True)

# 자모 시퀀스를 한글 음절로 조합한다.
generated_text = jamotools.join_jamos( generated_jamos)
print('\n\nGenerated sentence:\n')
print(generated_text)

# 시각화
plt.plot(loss_history, label='loss')
plt.legend()
plt.show()
