import sys

# 입력 받기
n = int(sys.stdin.readline())

# nC3 공식 적용 (n * (n-1) * (n-2) / 6)
# 정수 출력을 위해 // 연산자 사용
count = (n * (n - 1) * (n - 2)) // 6

print(count)
print(3)