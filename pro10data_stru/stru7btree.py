# 트리 구조 정의 (딕셔너리 활용)
tree = {
    'A': ('B', 'C'),
    'B': ('D', 'E'),
    'C': ('F', 'G'),
    'D': (None, None),
    'E': (None, None),
    'F': (None, None),
    'G': (None, None)
}

# ==========================================
# 1. 전위 순회 (Pre-order): 루트 -> 왼쪽 -> 오른쪽
# ==========================================
def preOrder(node):
    if node is None:
        return
    
    print(node, end=' ')      # 나를 먼저 출력!
    left, right = tree[node]
    preOrder(left)
    preOrder(right)

# ==========================================
# 2. 중위 순회 (In-order): 왼쪽 -> 루트 -> 오른쪽
# ==========================================
def inOrder(node):
    if node is None:
        return
    
    left, right = tree[node]
    inOrder(left)             # 왼쪽 끝까지 들어간 뒤
    print(node, end=' ')      # 중간에 나를 출력!
    inOrder(right)            # 그다음 오른쪽

# ==========================================
# 3. 후위 순회 (Post-order): 왼쪽 -> 오른쪽 -> 루트
# ==========================================
def postOrder(node):
    if node is None:
        return
    
    left, right = tree[node]
    postOrder(left)           # 왼쪽 끝까지 가고
    postOrder(right)          # 오른쪽 끝까지 간 뒤
    print(node, end=' ')      # 마지막에 나를 출력!


# ==========================================
# 실행 및 결과 확인
# ==========================================
print('전위 순회 (Pre-order) : ', end='')
preOrder('A') 
# 출력: A B D E C F G

print('\n중위 순회 (In-order)  : ', end='')
inOrder('A')  
# 출력: D B E A F C G

print('\n후위 순회 (Post-order): ', end='')
postOrder('A') 
# 출력: D E B F G C A
print()