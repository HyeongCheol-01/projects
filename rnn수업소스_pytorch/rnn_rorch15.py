실습15) IMDB 리뷰 감성 분류 : 2층 BiLSTM + Bahdanau Attention
!pip install torch datasets

import copy
import html
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)
from torch.utils.data import Dataset, DataLoader

# 1. 실행 환경 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("실행 장치:", device)

# 2. 주요 설정값
vocab_size = 10000
max_len = 500

embedding_dim = 128
hidden_size = 64
attention_units = 64

batch_size = 256
epochs = 3
learning_rate = 0.001

PAD_IDX = 0
UNK_IDX = 1


# 3. IMDB 데이터 불러오기
imdb_data = load_dataset("stanfordnlp/imdb")

# 테스트 데이터는 최종 평가에 사용하고,
# 훈련 데이터의 10%를 검증 데이터로 분리한다.
train_valid = imdb_data["train"].train_test_split(test_size=0.1, seed=42)
train_data = train_valid["train"]
valid_data = train_valid["test"]
test_data = imdb_data["test"]
print("훈련 데이터:", len(train_data))
print("검증 데이터:", len(valid_data))
print("테스트 데이터:", len(test_data))

# 4. 토큰화 및 단어 사전 생성
html_pattern = re.compile(r"<[^>]+>")
word_pattern = re.compile(r"[A-Za-z0-9']+")

def tokenize(text):
    """HTML 태그를 제거하고 영어 문장을 단어 단위로 분리한다."""
    text = html.unescape(text).lower()
    text = html_pattern.sub(" ", text)
    return word_pattern.findall(text)

counter = Counter()
review_lengths = []

# 단어 사전은 훈련 데이터만 사용하여 생성한다.
for text in train_data["text"]:
    tokens = tokenize(text)
    counter.update(tokens)
    review_lengths.append(len(tokens))

print("리뷰의 최대 길이:", max(review_lengths))
print("리뷰의 평균 길이:", sum(review_lengths) / len(review_lengths))

# 0번: Padding
# 1번: 사전에 없는 단어
word_to_index = { "<pad>": PAD_IDX, "<unk>": UNK_IDX}

# 가장 빈도가 높은 단어를 기준으로 최대 10,000개의 단어 사전을 만든다.
for word, _ in counter.most_common(vocab_size - 2):
    word_to_index[word] = len(word_to_index)

print("단어 사전 크기:",len(word_to_index))

def encode_review(text):
    """리뷰를 정수 시퀀스로 변환하고 길이를 500으로 맞춘다."""
    sequence = [
        word_to_index.get(word, UNK_IDX) for word in tokenize(text)
    ]

    # 긴 리뷰는 마지막 500개 토큰을 사용한다.
    sequence = sequence[-max_len:]

    if len(sequence) == 0:
        sequence = [UNK_IDX]

    actual_length = len(sequence)

    # pack_padded_sequence 사용을 위해 오른쪽 패딩
    sequence += [PAD_IDX] * ( max_len - actual_length )
    sequence = torch.tensor( sequence, dtype=torch.long)
    actual_length = torch.tensor( actual_length, dtype=torch.long)
    return sequence, actual_length


# 5. Dataset과 DataLoader
class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        sequence, length = encode_review(item["text"])
        label = torch.tensor(item["label"], dtype=torch.float32)
        return sequence, length, label

train_loader = DataLoader(
    IMDBDataset(train_data),batch_size=batch_size,shuffle=True,num_workers=0,
    pin_memory=torch.cuda.is_available()
)

valid_loader = DataLoader(
    IMDBDataset(valid_data), batch_size=batch_size, shuffle=False, num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    IMDBDataset(test_data), batch_size=batch_size, shuffle=False, num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# 6. 바다나우 어텐션
# LSTM의 마지막 은닉 상태만 사용하는 대신,모든 시점의 은닉 상태를 다시 참고한다.
# 현재 Query와 관련성이 높은 단어에는 더 높은 Attention Weight를 부여한다.
class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super().__init__()

        # BiLSTM 출력 차원:
        # forward 64 + backward 64 = 128
        self.W1 = nn.Linear(hidden_size * 2, units)
        self.W2 = nn.Linear(hidden_size * 2, units)

        self.V = nn.Linear(units, 1 )

    def forward(self, values, query, mask):
        # values: (batch_size, max_len, hidden_size * 2)
        # query: (batch_size, hidden_size * 2)

        # 모든 시점의 values와 더할 수 있도록 query의 시간축 차원을 추가한다.
        hidden_with_time_axis = query.unsqueeze(1)

        # score: (batch_size, max_len, 1)
        score = self.V(
            torch.tanh(self.W1(values) + self.W2(hidden_with_time_axis))
        )

        # 마지막 크기가 1인 차원 제거 (batch_size, max_len)
        score = score.squeeze(-1)

        # Padding 위치에는 Attention이 적용되지 않도록 매우 작은 값을 넣는다.
        score = score.masked_fill(
            ~mask,
            torch.finfo(score.dtype).min,
        )

        # 각 리뷰의 시간축을 기준으로 Softmax (batch_size, max_len)
        attention_weights = torch.softmax(score, dim=1 )

        # Attention Weight와 모든 시점의 LSTM 출력을 가중합한다.
        # attention_weights.unsqueeze(1): (batch_size, 1, max_len)
        # values: (batch_size, max_len, hidden_size * 2)
        # context_vector: (batch_size, hidden_size * 2)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return (context_vector,attention_weights )



# 7. 2층 BiLSTM + Attention 모델
class BiLSTMAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )

        # num_layers=2: 양방향 LSTM을 두 층으로 구성
        # bidirectional=True: 순방향과 역방향 LSTM 사용
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        self.attention = BahdanauAttention(attention_units )
        self.dense1 = nn.Linear( hidden_size * 2,  20 )

        self.dropout = nn.Dropout(0.5)

        # 이진 분류이므로 출력값은 1개
        self.output_layer = nn.Linear(20, 1 )

    def forward( self, inputs, lengths):
        # inputs: (batch_size, max_len)
        embedded = self.embedding(inputs)
        # embedded (batch_size, max_len, embedding_dim)

        # Padding이 LSTM상태계산에 영향주지않도록 실제리뷰길이를 이용 PackedSequence 생성
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False )

        packed_output, (
            hidden,
            cell,
        ) = self.lstm(packed)

        # 모든 시점의 LSTM 출력을
        # 다시 패딩된 텐서 형태로 복원
        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=inputs.size(1),
        )

        # lstm_output: (batch_size, max_len, hidden_size * 2)
        # hidden: (num_layers * num_directions,  batch_size,  hidden_size)

        hidden = hidden.reshape(
            2,                  # LSTM 층 개수
            2,                  # 순방향, 역방향
            inputs.size(0),     # batch_size
            hidden_size,
        )

        # 마지막 LSTM 층의 순방향 은닉 상태
        forward_h = hidden[-1, 0]

        # 마지막 LSTM 층의 역방향 은닉 상태
        backward_h = hidden[-1, 1]

        # 순방향과 역방향 은닉 상태 연결
        # state_h: (batch_size, hidden_size * 2)
        state_h = torch.cat(
            [
                forward_h,
                backward_h,
            ],  dim=1,
        )

        # 실제 단어는 True
        # Padding은 False
        attention_mask = (inputs != PAD_IDX)

        context_vector, attention_weights = (
            self.attention(
                values=lstm_output,
                query=state_h,
                mask=attention_mask,
            )
        )

        x = torch.relu( self.dense1(context_vector))
        x = self.dropout(x)

        # BCEWithLogitsLoss를 사용하므로 모델 내부에서는 Sigmoid를 적용하지 않는다.
        logits = self.output_layer(x).squeeze(1)
        return logits, attention_weights

model = BiLSTMAttentionModel().to(device)
print(model)

# 8. 손실 함수와 옵티마이저
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# 9. 학습 및 평가 함수
def run_epoch(data_loader, training=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct = 0
    total_count = 0

    for inputs, lengths, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, attention_weights = model(inputs, lengths)
            loss = criterion(logits, labels )

            if training:
                loss.backward()

                # LSTM에서 발생할 수 있는 기울기 폭주 방지
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        # 평가할 때만 Sigmoid를 적용하여 0~1 사이의 확률로 변환한다.
        probabilities = torch.sigmoid(logits)

        predictions = ( probabilities >= 0.5).float()
        total_loss += (loss.item() * labels.size(0) )
        total_correct += (predictions == labels).sum().item()
        total_count += labels.size(0)

    average_loss = (total_loss / total_count)
    accuracy = (total_correct / total_count)
    return average_loss, accuracy


# 10. 모델 학습
best_valid_loss = float("inf")
best_model_state = None

for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = run_epoch(train_loader, training=True )

    with torch.inference_mode():
        valid_loss, valid_accuracy = run_epoch(valid_loader,training=False)

    # 검증 손실이 가장 낮은 모델 저장
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model_state = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch}/{epochs} | "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Valid Loss: {valid_loss:.4f}, "
        f"Valid Acc: {valid_accuracy:.4f}"
    )


# 11. 테스트 데이터 평가
model.load_state_dict(best_model_state)

with torch.inference_mode():
    test_loss, test_accuracy = run_epoch(test_loader,training=False )

print(
    f"\n테스트 손실: "
    f"{test_loss:.4f}"
)

print(
    f"테스트 정확도: "
    f"{test_accuracy:.4f}"
)
