# 1. 노드 클래스 정의
class Node:
    def __init__(self, key):
        self.key = key      # 노드가 저장하는 값
        self.left = None    # 왼쪽 자식 노드
        self.right = None   # 오른쪽 자식 노드


# ==========================================
# [BST 삽입]
# 작으면 왼쪽, 크면 오른쪽으로 파고들어서 빈자리에 추가합니다.
# ==========================================
def insert(root, key):
    # 1. 현재 위치가 비어 있다면 새 노드를 생성해서 "반환"합니다.
    if root is None:
        return Node(key)

    # 2. 넣을 값이 현재 노드보다 작으면 왼쪽으로!
    if key < root.key:
        root.left = insert(root.left, key)
    # 3. 크거나 같으면 오른쪽으로!
    else:
        root.right = insert(root.right, key)
        
    # 4. 재귀적으로 연결을 유지하기 위해 자기 자신(root)을 반환합니다.
    return root


# ==========================================
# [중위 순회 (In-order)] : 왼쪽 -> 루트 -> 오른쪽
# ==========================================
def inorder(root, result):
    if root is None:
        return  # 더 내려갈 노드가 없다면 함수 탈출
    
    inorder(root.left, result)    # 1. 왼쪽 노드 방문
    result.append(root.key)       # 2. 현재 노드값 추가 (appen -> append 오타 수정)
    inorder(root.right, result)   # 3. 오른쪽 노드 방문 (누락된 부분 추가!)


# ==========================================
# 실행 부분
# ==========================================
# 무작위 순서로 데이터 삽입
values = [50, 30, 70, 20, 40, 60, 80]
root = None  # 처음엔 트리가 비어있음

for v in values:
    root = insert(root, v)  # BST에 삽입하고 루트(최상단)를 갱신 

print(f"삽입한 데이터 순서: {values}")

# 중위 순회 실행
sorted_result = []
inorder(root, sorted_result)
print(f"중위 순회(정렬) 결과: {sorted_result}")