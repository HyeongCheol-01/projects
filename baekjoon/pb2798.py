import sys
n, m = map(int, sys.stdin.readline().split())
# 카드에 적힌 숫자 리스트
cards = list(map(int, sys.stdin.readline().split()))

result = 0

# 3중 반복문을 이용한 완전 탐색
for i in range(n):
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            card_sum = cards[i] + cards[j] + cards[k]
            
            # 합이 M을 넘지 않는지 확인
            if card_sum <= m:
                # 그 중 가장 큰 값을 저장
                result = max(result, card_sum)

print(result)