실습11) 시퀀스-투-시퀀스(Sequence-to-Sequence, Seq2Seq)
# 입력 시퀀스를 인코더가 문맥 상태(hidden state, cell state)로 압축하고, 디코더가 해당 문맥 상태를 이용해 출력 시퀀스를 한 토큰씩 생성한다.
# 핵심 구조:
# Encoder -> Context States(h, c) -> Decoder
# 예:
# "나는 수학을 열심히 공부한다"
#       -> Encoder
#       -> 문맥 상태 (hidden state, cell state)
#       -> Decoder
#       -> "i study math hard"

import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


# 1. 재현성 및 실행 장치 설정
def set_seed(seed: int = 42) -> None:
    """실행할 때마다 가능한 한 동일한 결과가 나오도록 난수를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDA/cuDNN 환경에서 재현성을 높이기 위한 설정
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("실행 장치:", device)


# 2. 병렬 문장 데이터
# 한국어 입력 문장과 영어 출력 문장의 쌍이다.
# Seq2Seq 모델은 "한국어 입력 시퀀스 -> 영어 출력 시퀀스" 형태로 학습한다.
data = [
    ("안녕", "hi"),
    ("잘 지내?", "How are you?"),
    ("고마워", "thank you"),
    ("좋은 아침", "good morning"),
    ("사랑해", "i love you"),
    ("잘 자", "good night"),
]


# 3. 특수 토큰 및 토큰화 함수
PAD_TOKEN = "<pad>"   # 길이를 맞추기 위한 패딩 토큰
UNK_TOKEN = "<unk>"   # 학습 사전에 없는 단어
SOS_TOKEN = "<sos>"   # 출력 문장의 시작 토큰
EOS_TOKEN = "<eos>"   # 출력 문장의 종료 토큰


def tokenize(text: str) -> list[str]:
    """
    간단한 공백 단위 토큰화 함수. 영어는 소문자로 변환한다.
    filters=''를 사용한 기존 Keras Tokenizer와 비슷하게 물음표 등의 문장부호는 제거하지 않는다.
    """
    return text.lower().strip().split()


# 4. 단어 사전 클래스
class Vocabulary:
    """
    문장 목록으로부터 단어-정수 사전을 생성한다.
    word_to_index: 단어 -> 정수,  index_to_word: 정수 -> 단어
    """

    def __init__(self, texts: list[str], special_tokens: list[str]) -> None:
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}

        # 특수 토큰을 먼저 등록하여 고정된 번호를 부여한다.
        for token in special_tokens:
            self._add_token(token)

        counter = Counter()

        for text in texts:
            counter.update(tokenize(text))

        # 빈도가 높은 단어부터 등록한다.
        # 빈도가 같으면 알파벳/문자 순으로 정렬하여 결과를 일정하게 유지한다.
        sorted_tokens = sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )

        for token, _ in sorted_tokens:
            if token not in self.word_to_index:
                self._add_token(token)

    def _add_token(self, token: str) -> None:
        index = len(self.word_to_index)
        self.word_to_index[token] = index
        self.index_to_word[index] = token

    def encode(self, text: str) -> list[int]:
        """문장을 정수 시퀀스로 변환한다."""
        unk_index = self.word_to_index[UNK_TOKEN]

        encoded = [
            self.word_to_index.get(token, unk_index)
            for token in tokenize(text)
        ]

        # 빈 문자열이 들어오더라도 LSTM에 길이 0인 시퀀스가 전달되지 않게 한다.
        return encoded if encoded else [unk_index]

    def __len__(self) -> int:
        return len(self.word_to_index)


# 5. 입출력 문장 분리 및 단어 사전 생성
input_texts = [kor for kor, _ in data]
output_texts = [eng for _, eng in data]

# 인코더에는 PAD와 UNK 토큰이 필요하다.
encoder_vocab = Vocabulary( input_texts, special_tokens=[PAD_TOKEN, UNK_TOKEN],)

# 디코더에는 PAD, UNK, SOS, EOS 토큰이 필요하다.
decoder_vocab = Vocabulary(
    output_texts, special_tokens=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN],
)

ENC_PAD_IDX = encoder_vocab.word_to_index[PAD_TOKEN]
DEC_PAD_IDX = decoder_vocab.word_to_index[PAD_TOKEN]
SOS_IDX = decoder_vocab.word_to_index[SOS_TOKEN]
EOS_IDX = decoder_vocab.word_to_index[EOS_TOKEN]
print("\n인코더 단어 사전 : ", encoder_vocab.word_to_index)
print("\n디코더 단어 사전 : ", decoder_vocab.word_to_index)

# 6. 문장을 정수 시퀀스로 변환
encoder_sequences = [
    torch.tensor(encoder_vocab.encode(sentence), dtype=torch.long)
    for sentence in input_texts
]

# 디코더 전체 시퀀스: <sos> + 영어 문장 + <eos>
decoder_sequences = [
    torch.tensor([SOS_IDX] + decoder_vocab.encode(sentence) + [EOS_IDX], dtype=torch.long) for sentence in output_texts
]

# pack_padded_sequence에서 실제 문장 길이를 사용하기 위해 저장한다.
encoder_lengths = torch.tensor(
    [len(sequence) for sequence in encoder_sequences], dtype=torch.long,
)

# 7. 패딩 및 디코더 입력/정답 분리
# pad_sequence는 각 문장의 길이를 가장 긴 문장에 맞춘다.
encoder_input_data = pad_sequence(
    encoder_sequences,  batch_first=True, padding_value=ENC_PAD_IDX,
)

decoder_padded_data = pad_sequence(
    decoder_sequences, batch_first=True, padding_value=DEC_PAD_IDX,
)

# Teacher Forcing 학습 구조
# 디코더 입력: <sos> how are you?
# 디코더 정답: how are you? <eos>
decoder_input_data = decoder_padded_data[:, :-1]
decoder_target_data = decoder_padded_data[:, 1:]

print("\nencoder_input_data")
print(encoder_input_data)

print("\ndecoder_padded_data")
print(decoder_padded_data)

print("\ndecoder_input_data")
print(decoder_input_data)

print("\ndecoder_target_data")
print(decoder_target_data)

print("\n텐서 크기")
print("encoder_input_data :", encoder_input_data.shape)
print("decoder_input_data :", decoder_input_data.shape)
print("decoder_target_data:", decoder_target_data.shape)

# 8. Encoder
class Encoder(nn.Module):
    """
    입력 시퀀스를 읽고 마지막 hidden state와 cell state를 반환한다.
    PyTorch LSTM 상태 크기:
    hidden: (num_layers, batch_size, hidden_size)
    cell  : (num_layers, batch_size, hidden_size)
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_size: int, padding_idx: int,) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

    def forward(
        self, source: torch.Tensor, source_lengths: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        # source: (batch_size, source_sequence_length)
        embedded = self.embedding(source)

        # embedded:(batch_size, source_sequence_length, embedding_dim)
        # 패딩 위치가 LSTM의 최종 상태 계산에 영향을 주지 않도록
        # 실제 문장 길이를 이용해 PackedSequence로 변환한다.
        packed_embedded = pack_padded_sequence(
            embedded,
            lengths=source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # encoder_outputs는 사용하지않고, 마지막 hidden state와 cell state만 디코더에 전달한다.
        _, (hidden, cell) = self.lstm(packed_embedded)

        return hidden, cell


# 9. Decoder
class Decoder(nn.Module):
    """
    인코더의 상태를 초기 상태로 받아 출력 토큰의 logits를 생성한다.
    학습할 때는 정답 문장을 한 칸 이동한 decoder_input_data가 디코더 입력으로 사용된다.
    """

    def __init__(self, vocab_size:int, embedding_dim:int, hidden_size:int, padding_idx:int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

        # 각 시점의 LSTM 출력 벡터를 전체 단어 사전 크기의 logits로 변환한다.
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, target_input:torch.Tensor, hidden:torch.Tensor, cell:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # target_input: (batch_size, target_sequence_length)
        embedded = self.embedding(target_input)
        # initial_state에 해당하는 (hidden, cell)을 전달한다.
        decoder_outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # logits: (batch_size, target_sequence_length, decoder_vocab_size)
        # CrossEntropyLoss가 내부적으로 log-softmax를 처리하므로여기에서는 softmax를 적용 X
        logits = self.output_layer(decoder_outputs)
        return logits, hidden, cell

# 10. 전체 Seq2Seq 모델
class Seq2Seq(nn.Module):
    """Encoder와 Decoder를 하나의 학습용 모델로 연결한다."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source:torch.Tensor,source_lengths:torch.Tensor, target_input:torch.Tensor) -> torch.Tensor:
        # Encoder: 입력 문장 -> 문맥 상태(hidden, cell)
        hidden, cell = self.encoder(source, source_lengths)
        # Decoder: 문맥 상태 + 이전 정답 토큰들 -> 다음 토큰 logits
        logits, _, _ = self.decoder(target_input, hidden, cell)

        return logits


# 11. 모델 생성
embedding_dim = 64
hidden_size = 64

encoder = Encoder(
    vocab_size=len(encoder_vocab),
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    padding_idx=ENC_PAD_IDX,
)

decoder = Decoder(
    vocab_size=len(decoder_vocab),
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    padding_idx=DEC_PAD_IDX,
)

model = Seq2Seq(encoder, decoder).to(device)
print("\n모델 구조 : ", model)


# 12. DataLoader 생성
dataset = TensorDataset(
    encoder_input_data,
    encoder_lengths,
    decoder_input_data,
    decoder_target_data,
)

train_loader = DataLoader(dataset, batch_size=2, shuffle=True,)


# 13. 손실 함수 및 옵티마이저
# 패딩 토큰은 실제 정답이 아니므로 손실 계산에서 제외한다.
criterion = nn.CrossEntropyLoss(ignore_index=DEC_PAD_IDX)

# 예제 데이터가 매우 작으므로 비교적 큰 학습률을 사용한다.
# 실제 데이터에서는 보통 0.001 등의 값부터 조정한다.
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# 14. 모델 학습
epochs = 300

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for ( source_batch, source_length_batch, target_input_batch, target_batch) in train_loader:
        source_batch = source_batch.to(device)
        target_input_batch = target_input_batch.to(device)
        target_batch = target_batch.to(device)

        # source_length_batch는 pack_padded_sequence에서
        # 내부적으로 CPU 텐서가 필요하므로 CPU 상태로 둔다.
        optimizer.zero_grad(set_to_none=True)

        logits = model(source_batch, source_length_batch, target_input_batch)

        # logits: (batch_size, target_length, vocabulary_size)
        # target: (batch_size, target_length)
        # CrossEntropyLoss 계산을 위해 모든 시점을 하나의 축으로 펼친다.
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), target_batch.reshape(-1),
        )

        loss.backward()

        # 순환 신경망에서 발생할 수 있는 기울기 폭주를 완화한다.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0 )

        optimizer.step()
        total_loss += loss.item()

    if epoch == 1 or epoch % 50 == 0:
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {average_loss:.6f}")

print("\n학습 완료")


# 15. 번역 함수: Greedy Decoding
@torch.inference_mode()
def translate(sentence: str, max_length: int = 10) -> str:
    """
    입력 문장을 인코딩한 뒤 <sos>에서 시작하여 가장 확률이 높은 토큰을 한 개씩 선택한다.
    <eos>가 나오거나 max_length에 도달하면 생성을 종료한다.
    """
    model.eval()

    # 한국어 문장 -> 정수 시퀀스
    source_ids = encoder_vocab.encode(sentence)
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_length = torch.tensor([len(source_ids)], dtype=torch.long)

    # 입력 문장을 문맥 상태로 변환한다.
    hidden, cell = model.encoder( source_tensor, source_length)

    # 첫 번째 디코더 입력은 항상 <sos>이다.
    current_token = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)

    decoded_words: list[str] = []

    for _ in range(max_length):
        logits, hidden, cell = model.decoder( current_token, hidden, cell )

        # softmax를 계산하지 않아도 argmax 결과는 동일하다.
        next_token_index = int( logits[:, -1, :].argmax(dim=-1).item() )

        # 문장 종료 토큰이 나오면 생성을 멈춘다.
        if next_token_index == EOS_IDX:
            break

        next_word = decoder_vocab.index_to_word.get(next_token_index, UNK_TOKEN )

        # 특수 토큰은 최종 번역 문자열에 포함하지 않는다.
        if next_word not in { PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN }:
            decoded_words.append(next_word)

        # 방금 생성한 토큰을 다음 시점의 입력으로 사용한다.
        current_token = torch.tensor([[next_token_index]], dtype=torch.long, device=device )

    return " ".join(decoded_words)


# 16. 번역 테스트
print("\n번역 테스트")
for sentence in input_texts:
    translated = translate(sentence)
    print(f"{sentence:8s} ==> {translated}")
