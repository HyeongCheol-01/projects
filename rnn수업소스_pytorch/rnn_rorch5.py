실습5) 단어(공백으로 구분) 단위 자연어 생성 - 소설 토지 데이터 사용

import re
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/rnn_test_toji.txt'
path_to_file = Path('toji.txt')

# 파일이 없으면 다운로드
if not path_to_file.exists():
    urlretrieve(url, path_to_file)

with open(path_to_file, encoding='utf-8', errors='ignore') as obj:
    raw_text = obj.read()

print(raw_text[:100])
print('문자 수 : ', len(raw_text))

# 정제 후 corpus 만들기
def clean_str(text: str) -> str:
    text = re.sub(r"[^가-힣0-9() \n]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# print(clean_str('abc가나다   _^&$12하하'))

cleaned = clean_str(raw_text)
corpus = cleaned.replace("\n", " [NL] ")  # 줄바꿈을 하나의 토큰으로 처리

# 토큰 처리 : 문자열 -> 단어 분리 -> 단어 사전 -> 정수 번호로 변환
tokens = corpus.split()
word_counts = Counter(tokens)

# PAD=0, UNK=1로 설정하고 빈도수가 높은 단어부터 번호 부여
vocab = ['[PAD]', '[UNK]']
vocab += [word for word, count in word_counts.most_common()]

PAD, UNK = 0, 1
vocab_size = len(vocab)
tok2idx = { token: index for index, token in enumerate(vocab) }
print(f'어휘 수 : {vocab_size} (PAD={PAD}, UNK={UNK})')
print('샘플 어휘 : ', vocab[:20])

token_ids = np.array([tok2idx.get(token, UNK) for token in tokens], dtype=np.int64)
print('토큰 수 : ', len(token_ids))
print(token_ids)
# print(vocab[51], ' ', vocab[51341], ' ', vocab[2059])

if len(token_ids) <= 50:
    raise ValueError('토큰 수가 너무 적어 작업 안함')

# 학습용 시퀀스
SEQ_LEN = 15   # 과거 15개의 토큰을 보고 각 시점의 다음 토큰 예측
BATCH = 32     # 배치 크기
BUFFER = 2000  # PyTorch에서는 DataLoader의 shuffle=True 사용


# PyTorch Dataset은 학습 데이터를 한 개씩 꺼낼 수 있도록 관리하는 클래스
# DataLoader는 Dataset을 셔플하고 배치 단위로 전달한다.
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long )
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, index):
        # SEQ_LEN + 1 크기의 토큰 묶음 생성
        chunk = self.token_ids[index:index + self.seq_len + 1 ]

        x = chunk[:-1]   # 입력: 마지막 값 제외
        y = chunk[1:]    # 정답: 각 시점의 다음 토큰
        return x, y


dataset = TextDataset(token_ids, SEQ_LEN)
train_loader = DataLoader( dataset, batch_size=BATCH, shuffle=True, drop_last=True)

windows = len(token_ids) - SEQ_LEN

# 전체 학습 샘플을 배치 크기로 나누어 가능한 배치 수를 구한다.
# max(1, windows // BATCH) - 데이터가 적어 계산 결과가 0이 되더라도 최소 1번은 학습
steps_per_epoch = min(100, max(1, windows // BATCH))  
print('steps_per_epoch : ', steps_per_epoch)


# 모델
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
embedding_dim=128, padding_idx=PAD)

        # batch_first=True
        # 입력과 출력 형태: (batch, sequence, feature)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)

        self.dense1 = nn.Linear(256, 256)
        self.relu = nn.ReLU()

        # 각 시점에서 다음 단어의 로짓 출력
        self.dense2 = nn.Linear( 256, vocab_size )

    def forward(self, x):
        x = self.embedding(x)

        # output에는 모든 시점의 LSTM 출력이 저장됨
        output, (hidden, cell) = self.lstm(x)

        output = self.dense1(output)
        output = self.relu(output)

        # CrossEntropyLoss를 사용하므로 Softmax는 적용하지 않음
        logits = self.dense2(output)
        return logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용 장치 : ', device)

model = TextGenerationModel(vocab_size).to(device)
print(model)

# PAD에 해당하는 정답은 손실 계산에서 제외
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam( model.parameters())

# 모델이 예측한 로짓을 확률로 바꾼 뒤, 그 확률에 따라 다음 토큰 하나를 추출하는 함수
def sample_from_logits(logits, temperature=1.0, top_k=0, forbid_ids=(0, 1)):
    # 원본 로짓을 변경하지 않도록 복사
    logits = logits.detach().clone().float()

    # PAD, UNK 같은 생성하면 안 되는 토큰 제외
    for tid in forbid_ids:
        if 0 <= tid < logits.numel():
            logits[tid] = float('-inf')

    # temperature로 확률 분포 조절: 0에 가까우면 보수적이고 커질수록 다양한 단어가 선택됨
    if temperature <= 0:
        temperature = 1e-8

    logits = logits / temperature

    # top_k가 있으면 점수가 높은 상위 k개 후보만 사용
    if top_k:
        k = min(int(top_k), logits.numel())

        if 0 < k < logits.numel():
            # logits에서 값이 큰 상위 k개를 가져온다.
            # top_values: 상위 점수,  top_indices: 해당 단어 번호
            top_values, top_indices = torch.topk(logits, k)
            # logits와 같은 크기의 텐서를 만들고 모든 값을 -inf로 채운다.
            filtered_logits = torch.full_like(logits, float('-inf'))
            # 상위 k개 위치에만 원래 점수를 다시 넣는다.
            filtered_logits[top_indices] = top_values
            logits = filtered_logits   # 필터링된 결과를 최종 logits로 사용

    # 로짓을 확률로 변환
    probs = torch.softmax(logits, dim=0 )

    # 모든 확률이 비정상적인 경우를 방어
    if (not torch.isfinite(probs).all() or probs.sum().item() == 0 ):
        probs = torch.ones_like(probs)

        for tid in forbid_ids:
            if 0 <= tid < probs.numel():
                probs[tid] = 0

        probs = probs / probs.sum()

    # 확률에 따라 토큰 하나 선택
    return torch.multinomial( probs,num_samples=1).item()

idx2tok = np.array(vocab, dtype=object)

# 토큰 ID를 사람이 읽을 수 있는 문장으로 변환하는 함수
def ids_to_text(ids):
    # 예: ids=[2, 3, 5] -> ['사람', '간다', '나는']
    ids = np.array( ids, dtype=np.int64)
    toks = idx2tok[ids]

    # [NL] 토큰을 실제 줄바꿈으로 복원
    toks = ['\n' if token == '[NL]' else token for token in toks]
    return (' '.join(toks).replace(' \n ', '\n').replace(' \n', '\n').replace('\n ', '\n'))


# 사용자가 입력한 시작 문장을 바탕으로
# 학습된 모델이 뒤에 이어질 문장을 생성하는 함수
def generateFunc(seed_text, max_new_tokens=80, temperature=0.9, top_k=30):
    seed = clean_str(seed_text).replace('\n', ' [NL] ' )
    seed_tokens = seed.split()

    seed_ids = [tok2idx.get(token, UNK) for token in seed_tokens]

    # 입력 길이를 SEQ_LEN으로 맞춤. 부족한 부분은 왼쪽에 PAD를 추가
    context = ([PAD] * max(0, SEQ_LEN - len(seed_ids)) + seed_ids[-SEQ_LEN:])

    out_ids = []
    model.eval()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x = torch.tensor([context],  dtype=torch.long, device=device )
            # 마지막 시점의 로짓만 사용
            logits = model(x)[0, -1]
            tid = sample_from_logits(logits, 
temperature=temperature,top_k=top_k,forbid_ids=(PAD, UNK))
            out_ids.append(tid)

            # 가장 오래된 토큰을 제거하고 새로 생성한 토큰을 마지막에 추가
            context = context[1:] + [tid]

    text = ids_to_text(out_ids)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


EPOCHS = 2   # 실제 학습에서는 에폭 수를 더 크게 설정

history_loss = []
history_accuracy = []

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct_count = 0
    token_count = 0
    processed_steps = 0

    for step, (x_batch, y_batch) in enumerate(train_loader):
        if step >= steps_per_epoch:
            break

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)  # 이전에 계산된 기울기 초기화
        logits = model(x_batch)   # 예측

        # logits: (batch, sequence, vocab_size)
        # y_batch: (batch, sequence)
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )

        loss.backward()   # 역전파
        optimizer.step()   # 가중치 수정

        total_loss += loss.item()
        processed_steps += 1

        predicted = torch.argmax(logits, dim=-1 )

        mask = y_batch != PAD
        correct_count += ((predicted == y_batch) & mask).sum().item()
        token_count += mask.sum().item()

    epoch_loss = total_loss / processed_steps
    epoch_accuracy = correct_count / token_count
    history_loss.append(epoch_loss)
    history_accuracy.append(epoch_accuracy)

    print(
        f'Epoch {epoch + 1}/{EPOCHS} - '
        f'loss: {epoch_loss:.4f} - '
        f'accuracy: {epoch_accuracy:.4f}'
    )

    # 5에폭마다 또는 마지막 에폭에 샘플 출력
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        seed = '귀녀의 모습을 한번 쳐다보고 떠나려 했다.'
        sample = generateFunc(seed, max_new_tokens=80, temperature=0.9, top_k=30)
        print('\n[샘플 생성:', epoch + 1, ']')
        print(seed + ' ' + sample[:500])

print('final loss : ', history_loss[-1])
print('final acc : ', history_accuracy[-1])

# 최종 테스트
seed = '귀녀의 모습을 한번 쳐다보고 떠나려 했다.'

out = generateFunc(seed,max_new_tokens=100,temperature=0.8, top_k=40)
print('최종 결과 : \n')
print(seed + ' ' + out)
