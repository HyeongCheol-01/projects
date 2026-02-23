import sys

# 1. 숫자의 범위가 10,000까지라고 가정 (문제 조건을 확인하세요)
# 인덱스는 0부터 시작하므로 10001개의 칸을 만듭니다.
count = [0] * 10001

# 2. 첫 번째 줄에서 N 입력
n = int(sys.stdin.readline())

# 3. N번 반복하며 숫자를 입력받자마자 빈도수 체크
for _ in range(n):
    num = int(sys.stdin.readline())
    count[num] += 1  # 리스트에 추가하는 대신 해당 칸의 숫자만 올림

# 4. 빈도수 리스트를 돌며 숫자가 나타난 횟수만큼 출력
for i in range(10001):
    if count[i] != 0: # 숫자가 한 번이라도 나왔다면
        for _ in range(count[i]): # 나온 횟수만큼 반복 출력
            print(i)