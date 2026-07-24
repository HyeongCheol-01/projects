실습12)  Attention 개념 및 Q(Query)-K(Key)-V(Value) 구조 이해
# Dot-Product Attention 계산 과정
# 1. Query와 각 Key의 내적을 계산하여 유사도 점수 구하기
# 2. 점수에 Softmax를 적용하여 Attention Weight 구하기
# 3. Attention Weight를 각 Value에 곱한 뒤 합산하기
#
# Query: 현재 찾고 싶은 정보 또는 현재 상태
# Key  : Query와 비교할 각 입력 토큰의 특징
# Value: Attention Weight를 적용하여 실제로 가져올 정보

import torch

# 출력 시 소수점 표시 설정
torch.set_printoptions(precision=3, sci_mode=False )

# 1. Query, Key, Value 정의
# Query : 현재 디코더 상태 또는 현재 집중해서 찾고 싶은 정보
# 예: [달콤함, 새콤함]
Q = torch.tensor([2.0, 1.0], dtype=torch.float32 )

# Key : 인코더에서 만들어진 각 입력 토큰의 특징 벡터
# Key 개수: 3개
# 각 Key의 차원: 2차원
K = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

# Value : Attention Weight를 적용하여 실제로 가져올 정보
# 이 예제에서는 계산을 단순화하기 위해 Value를 Key와 같게 설정한다.
# 실제 Attention 모델에서는 Key와 Value가 서로 다른 벡터일 수 있다.
V = K.clone()
print("Q 크기:", Q.shape)   # torch.Size([2])
print("K 크기:", K.shape)   # torch.Size([3, 2])
print("V 크기:", V.shape)   # torch.Size([3, 2])

# 2. Attention Score 계산 : 각 Key와 Query의 내적을 계산한다.
# K: (3, 2)
# Q: (2,)
# 계산:
# 첫 번째 Key: [1, 0] · [2, 1] = 2
# 두 번째 Key: [0, 1] · [2, 1] = 1
# 세 번째 Key: [1, 1] · [2, 1] = 3
# 결과: scores = [2, 1, 3]
scores = torch.matmul(K, Q)

# @ 연산자를 사용해도 같은 결과
# scores = K @ Q
print("\nAttention Scores : ",  scores)

# 3. Softmax로 Attention Weight 계산
# Softmax를 이용해 점수를 합이 1인 확률 형태로 변환한다.
# dim=0: scores에 있는 3개의 값을 기준으로 Softmax를 계산한다.
weights = torch.softmax(scores, dim=0)
print("\nAttention Weights : ", weights)
print("Attention Weight 합:", weights.sum())
# tensor(1.)

# 4. 각 Value에 Attention Weight 적용
# weights 크기: (3,)
# V 크기: (3, 2)
# weights.unsqueeze(1)을 사용하여 weights를 (3,)에서 (3, 1)로 변경한다.
# 이렇게 해야 각 가중치가 해당 Value 벡터 전체에 곱해진다.
weighted_values = weights.unsqueeze(dim=1) * V
print("\n가중치가 적용된 Value : ", weighted_values)

# 5. 가중합을 계산하여 최종 Attention 출력 생성
# 각 Value 벡터를 행 방향으로 더한다.
# weighted_values: (3, 2)
# output         : (2,)
output = weighted_values.sum(dim=0)
print("\nAttention 최종 출력", output)
# 6. 행렬곱으로 간단하게 계산. 다음 계산은 위의 가중합과 동일하다.
# weights: (3,)
# V      : (3, 2)
# 결과: output_by_matmul: (2,)

output_by_matmul = torch.matmul(weights, V)

# @ 연산자를 사용해도 동일.   output_by_matmul = weights @ V
print("\n행렬곱으로 계산한 Attention 출력 : ", output_by_matmul)

# 두 계산 결과가 같은지 검증
print("두 계산 결과가 같은가?", torch.allclose(output, output_by_matmul))

# 7. Attention 계산 과정을 직접 확인 : 반올림한 값을 직접 입력하지 않고, 실제 계산된 weights 값을 사용한다. 이렇게 해야 반올림으로 인한 계산 오차를 방지할 수 있다.
out1 = (weights[0] * V[0, 0] + weights[1] * V[1, 0] + weights[2] * V[2, 0])
out2 = (weights[0] * V[0, 1] + weights[1] * V[1, 1] + weights[2] * V[2, 1])
manual_output = torch.stack([out1, out2])
print("\n직접 계산한 결과", manual_output)

# 8. 최종 결과 정리
print("\n최종 결과 =====")
print("Scores:", torch.round(scores, decimals=3))
print("Weights:",torch.round(weights, decimals=3))
print("Output:",torch.round(output, decimals=3))
