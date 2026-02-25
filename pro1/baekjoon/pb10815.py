import sys


n = int(sys.stdin.readline())
sanggeun_cards = set(sys.stdin.readline().split())

m = int(sys.stdin.readline())
check_cards = sys.stdin.readline().split()

# 2. 결과 확인
result = []
for card in check_cards:
    if card in sanggeun_cards:
        result.append('1')
    else:
        result.append('0')

# 3. 출력
print(" ".join(result))