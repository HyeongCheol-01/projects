실습13) 번역 Dot-Product Attention 모델 작성
# "I love you" -> "난 너를 사랑해"

import numpy as np
import math

src_tokens = ["I", "love", "you"]
tgt_tokens = ["난", "너를", "사랑해"]

n_src = len(src_tokens)
K = np.eye(n_src)
V = np.eye(n_src)

n_tgt = len(tgt_tokens)
Q = np.zeros((n_tgt, n_src))

# 출력 단어(tgt_tokens) 하나하나에 그 단어가 어느 입력 단어(src_tokens)에 집중할 지를 설정
for i in range(n_tgt):
    if i == 0:   # 출력 첫단어(i==0)는 입력 첫단어(src[0])에 100% 집중
        Q[i, 0] = 1.0
    elif i == n_tgt - 1:
        Q[i, -1] = 1.0
    else:
        Q[i, i:i + 2] = 0.5  # 중간 출력은 두 입력에 분산 집중

print(Q)

def scaled_atten_func(q, K, V):
    scores = q.dot(K.T) / math.sqrt(K.shape[1])  # 유사도 점수 계산 / 스케일링
    exp = np.exp(scores - np.max(scores))    # softmax 가중치
    weights = exp / exp.sum()    # 가중치 합 (Attention weight)
    context = (weights[:, None] * V).sum(axis=0)   # 동적 컨텍스트벡터
    print('context : ', context)
    return context, weights

# 각 목표 단어 마다 어텐션 실행 후 결과 확인
generated = []  # 생성된 단어 저장용

for i, tgt in enumerate(tgt_tokens):
    context, weights = scaled_atten_func(Q[i], K, V)
    print(f'생성 단어 : {tgt}')
    print('입력 단어별 어텐션 가중치:')
    for src_token, w in zip(src_tokens, weights):
        print(f'{src_token:>7}:{w:.3f}')
    print()
    generated.append(tgt)

print('최종 번역 결과')
print(' '.join(generated))
