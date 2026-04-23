#데큐(Deque) : 양쪽 끝에서 삽입과 삭제가 모두 가능한 자료구조
# 놀이공원 우선 탑승 + 일반 대기 줄

from collections import deque 

dq = deque()
print('놀이공원 대기 시작')

# 일반인은 뒤쪽으로 들어옴(Queue 처럼)