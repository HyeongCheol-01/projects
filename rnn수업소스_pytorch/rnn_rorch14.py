실습14) Scaled Dot-Product Attention (Tokenizer + Attention)
import math
from collections import Counter
import torch
import torch.nn.functional as F

# 간단한 토크나이저 함수
# PyTorch에는 Keras Tokenizer와 동일한 기본 클래스가 없으므로
# 공백 단위 토큰화를 간단한 함수로 구현한다.
def tokenizerFunc(text):
    words = text.split()
    word_counts = Counter(words)

    # 단어 인덱스는 1부터 시작한다. 0은 padding 값으로 사용한다.
    word_index = {word: idx for idx, word in enumerate(words, start=1)}
    index_word = {idx: word for word, idx in word_index.items()}
    sequence = [ word_index[word] for word in words]

    return sequence, word_index, word_counts, index_word

# 입출력 문장
input_text = "I have a pen"
output_text = "<sos> 나는 펜을 갖고 있다 <eos>"

# 토큰 처리
input_seq, input_word_index, input_word_counts, input_index_word = (
    tokenizerFunc(input_text)
)
output_seq, output_word_index, output_word_counts, output_index_word = (
    tokenizerFunc(output_text)
)
print("input_word_index :", input_word_index)
print("input_word_counts:", input_word_counts)
print("output_word_index:", output_word_index)

# 문장을 숫자 시퀀스로 변환. batch 차원을 포함하여 2차원 텐서로 만든다.
input_seq = torch.tensor([input_seq], dtype=torch.long )
output_seq = torch.tensor([output_seq], dtype=torch.long )
print("input_seq :", input_seq)
print("output_seq:", output_seq)

# 시퀀스 패딩
max_input_len = input_seq.size(1)
max_output_len = output_seq.size(1)
print(max_input_len, max_output_len)

# 현재는 문장이 각각 하나이므로 추가 패딩이 발생하지 않는다.
# 부족한 길이가 있으면 오른쪽에 0을 채우는 방식이다.
input_pad = F.pad( input_seq, pad=(0, max_input_len - input_seq.size(1)), value=0)
output_pad = F.pad( output_seq, pad=(0, max_output_len - output_seq.size(1)), value=0)

# Q, K, V 구성
n_src = input_pad.shape[1]    # 입력 시퀀스 길이
n_tgt = output_pad.shape[1]   # 출력 시퀀스 길이

# 입력 위치를 구분하기 위해 단위행렬 사용
K = torch.eye(n_src)
V = K.clone()

# 개념 이해용 예제이므로 Query가 어느 입력 위치에 집중할지 직접 지정한다.
Q = torch.zeros((n_tgt, n_src))

for i in range(n_tgt):
    if i == 0: Q[i, 0] = 1.0
    elif i == n_tgt - 1: Q[i, -1] = 1.0
    elif i < n_src - 1: Q[i, i:i + 2] = 0.5
    else: Q[i, -1] = 1.0

# print(Q)

# Scaled Dot-Product Attention 함수
def attentionFunc(q, K, V):
    # Query와 모든 Key의 내적 계산
    # q   : (n_src,)
    # K.T : (n_src, n_src)
    # scores: (n_src,)
    scores = torch.matmul(q, K.T)

    # Key 벡터 차원의 제곱근으로 나누어 스케일링
    scores = scores / math.sqrt(K.shape[1])

    # 각 입력 위치에 대한 Attention Weight
    weights = torch.softmax(scores, dim=0)

    # Value의 가중합으로 Context Vector 생성
    context = torch.matmul(weights, V)
    return context, weights


# Attention 실행 : 디코더가 한 스텝씩 출력하는 과정
print("\nAttention 결과\n")

# Attention Weight는 입력 시퀀스의 각 위치에 대응한다.
input_words = input_text.split()

for i in range(n_tgt):
    context, weights = attentionFunc(Q[i], K, V )
    print(f"[Output 위치 {i}]")

    for src_word, weight in zip(input_words, weights):
        print(
            f"  - {src_word:>5} "
            f"-> Attention:{weight.item():.3f}"
        )

    print("  -> Context 벡터:", torch.round(context * 1000) / 1000, "\n")

# 최종 출력 시퀀스
print("입력 시퀀스 :", input_pad)
print("출력 시퀀스 :", output_pad)

# 디코더 출력 복원 : 0, <sos>, <eos> 토큰 제외
print(output_index_word)
reconstructed = []

for idx in output_pad[0]:
    idx = idx.item()
    if idx == 0: continue

    word = output_index_word.get( idx, "?" )
    if word in ["<sos>", "<eos>"]: continue

    reconstructed.append(word)


print("\n복원 결과\n")
print(" ".join(reconstructed))
