실습5 - 1)  토지 예제 2

import re
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 2. 데이터 파일 다운로드
url = (
    'https://raw.githubusercontent.com/pykwon/python/'
    'refs/heads/master/testdata_utf8/rnn_test_toji.txt'
)
path_to_file = Path('toji.txt')

# 파일이 존재하지 않을 때만 다운로드한다.
if not path_to_file.exists():
    urlretrieve(url, path_to_file)

# 텍스트 파일 읽기
with open(path_to_file, encoding='utf-8', errors='ignore') as obj:
    raw_text = obj.read()

print(raw_text[:100])
print('문자 수 : ', len(raw_text))

# 텍스트 정제 함수 정의
def clean_str(text: str) -> str:
    text = re.sub(r"[^가-힣0-9() \n]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)     # 공백이 2개 이상 반복되면 하나로 줄인다.
    return text.strip()                  # 앞뒤 공백을 제거해서 반환

# 말뭉치 만들기
cleaned = clean_str(raw_text)                  # 원문 텍스트를 정제한다.
corpus = cleaned.replace("\n", " [NL] ")       # 줄바꿈을 [NL] 토큰으로 바꾼다.
# 6. 주요 설정값 지정
MAX_TOKENS = 3000     # 출력층 크기를 줄여 학습 속도를 높인다.
SEQ_LEN = 15           # 문맥 길이를 줄여 LSTM 계산량을 줄인다.
BATCH = 32             # 배치 크기를 줄여 메모리 부담을 낮춘다.
EMBED_DIM = 64        # 임베딩 차원을 줄인다.
LSTM_UNITS = 128       # LSTM 은닉 상태 크기를 줄인다.
EPOCHS = 2             # 짧게 학습한다.

tokens = corpus.split()    # 공백을 기준으로 단어 분리

word_counts = Counter(tokens)  #  단어 사전 생성
vocab = ['[PAD]', '[UNK]']  # 0번은 PAD, 1번은 UNK로 사용한다.

# 자주 등장하는 단어 위주로 최대 MAX_TOKENS개까지 사용한다.
normal_words = [
    word for word, count in word_counts.most_common() if word not in ('[PAD]', '[UNK]')
]

vocab.extend(normal_words[:MAX_TOKENS - 2])

# 9. 단어 사전 확인
PAD = 0    # 0번 토큰은 패딩 토큰이다.
UNK = 1    # 1번 토큰은 사전에 없는 단어 토큰이다.

vocab_size = len(vocab)

# 단어를 토큰 ID로 변환하기 위한 사전
tok2idx = {token: index for index, token in enumerate(vocab)}
print(f'어휘 수 : {vocab_size} (PAD={PAD}, UNK={UNK})')
print('샘플 어휘 : ', vocab[:20])

# 10. 전체 corpus를 토큰 ID 시퀀스로 변환
token_ids = np.array([tok2idx.get(token, UNK) for token in tokens], dtype=np.int64)
print('토큰 수 : ', len(token_ids))
print(token_ids[:20])
print(vocab[token_ids[0]])
print(vocab[token_ids[1]])
print(vocab[token_ids[2]])

# 11. 데이터 수 확인
if len(token_ids) <= SEQ_LEN + 1:
    raise ValueError('토큰 수가 너무 적어 학습할 수 없다.')

# 12. Dataset 클래스 정의
class TojiDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = torch.tensor(token_ids,dtype=torch.long )
        self.seq_len = seq_len

    def __len__(self):
        # 만들 수 있는 전체 슬라이딩 윈도우 개수
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, index):
        # 현재 위치부터 SEQ_LEN개의 토큰을 입력으로 사용한다.
        x = self.token_ids[index:index + self.seq_len]

        # 입력보다 한 칸 뒤의 토큰들을 정답으로 사용한다.
        y = self.token_ids[index + 1:index + self.seq_len + 1 ]
        return x, y

dataset = TojiDataset( token_ids, SEQ_LEN)

# 13. DataLoader 생성
train_loader = DataLoader(
    dataset,
    batch_size=BATCH,    # 지정한 크기만큼 배치로 묶는다.
    shuffle=True,         # 학습할 때 데이터 순서를 섞는다.
    drop_last=True       # 크기가 BATCH보다 작은 마지막 배치는 제외한다.
)

# 14. 입력과 정답 형태 확인
sample_x, sample_y = dataset[0]
print('입력 샘플 : ', sample_x)
print('정답 샘플 : ', sample_y)
print('입력 크기 : ', sample_x.shape)
print('정답 크기 : ', sample_y.shape)

windows = len(token_ids) - SEQ_LEN  # 15. 학습 스텝 수 계산

# 한 epoch에서 사용할 배치 횟수를 최대 100번으로 제한한다.
steps_per_epoch = min(100, max(1, windows // BATCH))
print('steps_per_epoch : ', steps_per_epoch)

# 16. 모델 생성
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 입력 가능한 토큰 ID 개수
            embedding_dim=embed_dim,    # 각 단어를 지정한 차원의 벡터로 변환
            padding_idx=PAD             # PAD 토큰의 임베딩 벡터는 0으로 유지
        )

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_units, batch_first=True)
        self.dense1 = nn.Linear(lstm_units, 256 )
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear( 256,  vocab_size )

    def forward(self, x):
        # 입력 형태: (batch, sequence)
        x = self.embedding(x)

        # 임베딩 결과: (batch, sequence, embed_dim)
        output, (hidden, cell) = self.lstm(x)

        # 각 시점의 LSTM 출력에 완전연결층을 적용한다.
        output = self.dense1(output)
        output = self.relu(output)

        # 각 시점마다 전체 단어 후보에 대한 점수를 출력한다.
        # CrossEntropyLoss를 사용하므로 Softmax는 적용하지 않는다.
        logits = self.dense2(output)
        return logits

# GPU가 있으면 GPU를 사용하고, 없으면 CPU를 사용한다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용 장치 : ', device)

model = TextGenerationModel(
    vocab_size=vocab_size, embed_dim=EMBED_DIM, lstm_units=LSTM_UNITS).to(device)
print(model)

# 17. 손실 함수와 최적화 알고리즘 설정
# CrossEntropyLoss는 정수 형태의 정답 토큰 ID와 logits를 사용한다.
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters())

# 18. ID를 토큰으로 바꾸기 위한 배열 생성
idx2tok = np.array(vocab, dtype=object)

# 19. 토큰 ID 목록을 텍스트로 바꾸는 함수 정의
def ids_to_text(ids):
    # 토큰 ID들을 실제 단어로 바꾼다.
    toks = idx2tok[np.array(ids, dtype=np.int64)]

    # [NL] 토큰은 실제 줄바꿈으로 복원한다.
    toks = ['\n' if token == '[NL]' else token for token in toks]
    # 단어 사이에 공백을 넣어 하나의 문자열로 만든다.
    text = ' '.join(toks)

    # 줄바꿈 앞뒤의 불필요한 공백을 정리한다.
    text = text.replace(' \n ', '\n')
    text = text.replace(' \n', '\n')
    text = text.replace('\n ', '\n')
    return text

# 20. logits에서 다음 토큰 하나를 샘플링하는 함수 정의
def sample_from_logits(logits, temperature=1.0, top_k=30, forbid_ids=(0, 1)):
    # 원본 logits를 변경하지 않도록 복사한다.
    logits = logits.detach().clone().to(dtype=torch.float64 )
    # 생성하면 안 되는 토큰 ID를 하나씩 확인한다.
    for token_id in forbid_ids:
        if 0 <= token_id < logits.numel():
            # PAD와 UNK가 선택되지 않도록 점수를 -무한대로 바꾼다.
            logits[token_id] = float('-inf')

    # temperature가 0 이하이면 0으로 나누는 문제가 발생한다.
    if temperature <= 0:
        temperature = 1e-8

    # temperature로 확률 분포를 조절한다.
    logits = logits / temperature

    # top_k가 지정되어 있으면 상위 후보만 남긴다.
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.numel() )

        if k < logits.numel():
            # 점수가 높은 상위 k개의 점수와 토큰 ID를 가져온다.
            top_values, top_indices = torch.topk(logits,  k)

            # 모든 토큰의 점수를 -무한대로 채운다.
            filtered_logits = torch.full_like(logits,float('-inf'))

            # 상위 k개 토큰의 점수만 원래 값으로 복원한다.
            filtered_logits[top_indices] = top_values

            logits = filtered_logits

    # logits를 확률값으로 변환한다.
    probs = torch.softmax(logits,dim=0)

    # 확률값에 문제가 발생한 경우 균등 확률로 대체한다.
    if (not torch.isfinite(probs).all() or probs.sum().item() == 0):
        probs = torch.ones_like(probs)

        for token_id in forbid_ids:
            if 0 <= token_id < probs.numel():
                probs[token_id] = 0

        probs = probs / probs.sum()

    # 확률 분포에 따라 다음 토큰 ID를 하나 선택한다.
    return torch.multinomial( probs, num_samples=1).item()


# 21. 문장 생성 함수 정의
def generateFunc(seed_text, max_new_tokens=80, temperature=0.9, top_k=30):
    seed = clean_str(seed_text)   # 시작 문장을 정제한다.
    seed = seed.replace('\n', ' [NL] ')   # 시작 문장의 줄바꿈을 [NL] 토큰으로 바꾼다.
    seed_tokens = seed.split()  # 시작 문장을 토큰 단위로 분리한다.

    # 각 단어를 토큰 ID로 변환한다.
    seed_ids = [tok2idx.get(token, UNK) for token in seed_tokens ]
    # 시작 문장이 짧으면 왼쪽을 PAD로 채운다.
    context = [ PAD ] * max(0, SEQ_LEN - len(seed_ids))

    # 최근 SEQ_LEN개의 토큰만 문맥으로 사용한다.
    context = context + seed_ids[-SEQ_LEN:]

    out_ids = []  # 생성된 토큰 ID를 저장한다.
    model.eval()  # 평가 모드로 변경한다.

    # 문장 생성 시에는 기울기를 계산하지 않는다.
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # 현재 문맥을 모델 입력 텐서로 변환한다.
            x = torch.tensor([context], dtype=torch.long, device=device )

            pred = model(x)   # 현재 문맥 다음에 올 단어 점수를 예측한다.
            logits = pred[0, -1]   # 마지막 시점의 예측 결과만 가져온다.

            # 다음 토큰 ID를 샘플링한다.
            next_id = sample_from_logits(logits, 
temperature=temperature,top_k=top_k,forbid_ids=(PAD, UNK))

            # 생성된 토큰 ID를 결과 리스트에 추가한다.
            out_ids.append(next_id)

            # 가장 오래된 토큰을 빼고 새 토큰을 문맥에 추가한다.
            context = context[1:] + [next_id]

    text = ids_to_text(out_ids)   # 생성된 토큰 ID들을 텍스트로 바꾼다.
    text = re.sub(r"[^\S\n]{2,}", " ", text).strip()  # 여러 공백을 하나로 정리한다.
    return text


# 22. 학습 중 샘플 문장을 출력하는 함수 정의
def print_sample(current_epoch, total_epochs, sample_every=5):
    # 지정한 epoch가 아니고 마지막 epoch도 아니면 출력하지 않는다.
    if (current_epoch % sample_every != 0 and current_epoch != total_epochs ):
        return

    # 샘플 생성을 위한 시작 문장을 지정한다.
    seed = '귀녀의 모습을 한번 쳐다보고 떠나려 했다.'
    sample = generateFunc( seed, max_new_tokens=80,temperature=0.9, top_k=30 )
    print(f'\n[샘플 생성: {current_epoch} epoch]')
    print(seed + ' ' + sample[:500])

# 23. 모델 학습
history_loss = []
history_acc = []

for epoch in range(EPOCHS):
    model.train()  # 학습 모드로 변경한다.
    total_loss = 0.0
    correct_count = 0
    token_count = 0
    processed_steps = 0

    for step, (x_batch, y_batch) in enumerate(train_loader):
        # 한 epoch에서 지정한 step 수만큼만 학습한다.
        if step >= steps_per_epoch:
            break

        # 입력과 정답을 연산 장치로 이동한다.
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 이전에 계산된 기울기를 초기화한다.
        optimizer.zero_grad(set_to_none=True)

        # 다음 토큰에 대한 logits를 예측한다.
        logits = model(x_batch)

        # logits 형태: (batch, sequence, vocab_size)
        # 정답 형태: (batch, sequence)

        # 모든 배치와 시점을 한 줄로 펼쳐 손실을 계산한다.
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )

        loss.backward()  # 역전파를 수행한다.
        optimizer.step() # 모델의 가중치를 수정한다.

        total_loss += loss.item()
        processed_steps += 1

        # 각 시점에서 점수가 가장 높은 토큰 ID를 예측값으로 선택한다.
        predicted = torch.argmax(logits, dim=-1)
        mask = y_batch != PAD  # PAD가 아닌 정답만 정확도 계산에 사용한다.

        correct_count += ((predicted == y_batch) & mask).sum().item()
        token_count += mask.sum().item()

    # 현재 epoch의 평균 손실을 계산한다.
    epoch_loss = total_loss / processed_steps

    # 현재 epoch의 정확도를 계산한다.
    epoch_acc = correct_count / token_count
    history_loss.append(epoch_loss)
    history_acc.append(epoch_acc)
    print(
        f'Epoch {epoch + 1}/{EPOCHS} - '
        f'loss: {epoch_loss:.4f} - '
        f'acc: {epoch_acc:.4f}'
    )

    # 지정한 주기마다 샘플 문장을 출력한다.
    print_sample(current_epoch=epoch + 1, total_epochs=EPOCHS,  sample_every=5 )

# 24. 최종 학습 결과 확인
print('final loss : ', history_loss[-1])
print('final acc : ', history_acc[-1])

# 25. 최종 문장 생성 테스트
seed = '귀녀의 모습을 한번 쳐다보고 떠나려 했다.'
out = generateFunc( seed,  max_new_tokens=200, temperature=0.8, top_k=50)
print('최종 결과 : \n')
print(seed + ' ' + out)
