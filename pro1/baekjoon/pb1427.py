import sys

# 1. 숫자를 문자열 형태로 입력받음
n = sys.stdin.readline().strip()

# 2. 문자열의 각 문자를 리스트의 요소로 만듦 (예: '2143' -> ['2', '1', '4', '3'])
digits = list(n)

# 3. 내림차순 정렬 (큰 숫자가 앞으로)
digits.sort(reverse=True)

# 4. 리스트 요소를 하나의 문자열로 합쳐서 출력
print("".join(digits))