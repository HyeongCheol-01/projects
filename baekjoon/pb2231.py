
import sys

# 자연수 N 입력받기
n = int(sys.stdin.readline())

result = 0

# 1부터 n까지 모든 숫자를 확인 (완전 탐색)
for i in range(1, n + 1):
    # 각 자릿수의 합 계산 (예: 245 -> 2+4+5)
    digit_sum = sum(map(int, str(i)))
    
    # 분해합 = 원래 숫자 + 자릿수의 합
    target_sum = i + digit_sum
    
    # 만약 분해합이 n과 같다면 i는 n의 생성자
    if target_sum == n:
        result = i
        break  # 가장 작은 생성자를 구해야 하므로 바로 탈출

print(result)

