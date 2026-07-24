실습16) 트렌스포머 Self-Attention 학습 예제

# "it"이 "animal"을 참고하도록 Attention 학습

import math
import torch
import torch.nn as nn

# 실행 결과 재현을 위한 난수 고정
torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)

# 입력 문장
tokens = [
    "The",
    "animal",
    "didn't",
    "cross",
    "the",
    "street",
    "because",
    "it",
    "was",
    "too",
    "tired",
    "."
]


# 단어 사전 생성
vocab = list(dict.fromkeys(tokens))
print("vocab:", vocab)

word_to_id = { word: i for i, word in enumerate(vocab) }
print("word_to_id:", word_to_id)
id_to_word = { i: word for word, i in word_to_id.items() }
print("id_to_word:", id_to_word)

# 문장을 정수 인덱스 Tensor로 변환
input_ids = torch.tensor([word_to_id[word] for word in tokens], dtype=torch.long)
print("input_ids:", input_ids)

# 문장에서 it과 animal의 위치
it_pos = tokens.index("it")
animal_pos = tokens.index("animal")
print("it 위치:", it_pos)
print("animal 위치:", animal_pos)

# Self-Attention 모델
class MySelfAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        # 단어 인덱스를 임베딩 벡터로 변환
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # 임베딩 벡터를 Query, Key, Value로 변환
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, input_ids, query_pos):
        # 입력 단어 임베딩   x: (sequence_length, embed_dim)
        x = self.embedding(input_ids)

        # Query, Key, Value 생성
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 전체 토큰 간 Scaled Dot-Product Attention Score
        # Q   : (sequence_length, embed_dim)
        # K.T : (embed_dim, sequence_length)
        # scores: (sequence_length, sequence_length)
        scores = torch.matmul(Q, K.transpose(0, 1))
        scores = scores / math.sqrt(K.size(-1))

        # "it" 위치의 Query가 모든 Key를 본 점수
        # clone()으로 복사한 후 마스크를 적용해야 원본 scores를 직접 수정하지 않아 안전하다.
        query_scores = scores[query_pos].clone()

        # it이 자기 자신을 참고하지 못하도록 마스크 적용
        query_scores[query_pos] = torch.finfo(query_scores.dtype).min

        # 각 단어에 대한 Attention Weight
        attention_weights = torch.softmax(query_scores, dim=0)

        # Attention Weight와 Value의 가중합
        context_vector = torch.matmul( attention_weights, V)

        return (query_scores, attention_weights, context_vector, x, Q, K, V )


# 모델 생성
vocab_size = len(vocab)
embed_dim = 8

model = MySelfAttention(vocab_size=vocab_size, embed_dim=embed_dim)

# 학습 전 결과
model.eval()

with torch.no_grad():
    (
        _,
        weights_before,
        _,
        embeddings_before,
        _,
        _,
        _
    ) = model(
        input_ids,
        query_pos=it_pos
    )


print("\n[학습 전 임베딩 일부]")
print("animal embedding:", embeddings_before[animal_pos])
print("it embedding:", embeddings_before[it_pos])

print("\n[학습 전 it의 Attention]")
for token, weight in zip(tokens, weights_before):
    print(
        f"{token:>8} "
        f"-> attention weight: {weight.item():.4f}"
    )


# 모델 학습
optimizer = torch.optim.Adam(model.parameters(),lr=0.05)

# 정답은 animal이 있는 위치
target = torch.tensor([animal_pos], dtype=torch.long)

# query_scores는 Softmax 적용 전 logits
loss_fn = nn.CrossEntropyLoss()

model.train()

for epoch in range(500):
    # 이전 기울기 초기화
    optimizer.zero_grad()
    (
        query_scores,
        attention_weights,
        context_vector,
        x,
        Q,
        K,
        V
    ) = model(input_ids, query_pos=it_pos)

    # CrossEntropyLoss 입력:
    # query_scores: (1, sequence_length)
    # target      : (1,)
    loss = loss_fn( query_scores.unsqueeze(0), target)

    # 자동 미분
    loss.backward()

    # 가중치 갱신
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"epoch {epoch:3d} | "
            f"loss: {loss.item():.5f}",
            flush=True
        )


# 학습 후 결과
model.eval()

with torch.no_grad():
    (
        query_scores,
        attention_weights,
        context_vector,
        embeddings_after,
        Q,
        K,
        V
    ) = model(input_ids, query_pos=it_pos)


print("\n[학습 후 임베딩 일부]")
print("animal embedding:", embeddings_after[animal_pos])
print("it embedding:", embeddings_after[it_pos])

print("\n[학습 후 it의 Attention]")
for token, score, weight in zip(tokens, query_scores, attention_weights):
    print(
        f"{token:>8} "
        f"-> score: {score.item():9.4f}, "
        f"attention weight: {weight.item():.4f}"
    )

max_index = torch.argmax(attention_weights).item()

print("\n[결론]")
print(
    f"'it'이 가장 많이 참고한 단어는 "
    f"'{tokens[max_index]}'"
)
print("it의 context vector:", context_vector)
