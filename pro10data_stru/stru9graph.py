from collections import deque

# 1. 그래프 정의 (딕셔너리 활용)
# 키(Key)는 노드, 값(Value)은 연결된 이웃 노드들의 리스트입니다.
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# ==========================================
# [DFS] 깊이 우선 탐색 (재귀 함수 사용)
# ==========================================
def dfsFunc(graph, node, visited=None):
    # 처음 호출될 때 방문 리스트 초기화
    if visited is None:
        visited = []
        
    # 1. 현재 노드를 방문 처리
    visited.append(node)
    
    # 2. 현재 노드와 연결된 이웃 노드들을 하나씩 확인
    for neighbor in graph[node]:
        # 아직 방문하지 않은 이웃이라면 그곳으로 깊게 파고듦(재귀)
        if neighbor not in visited:
            dfsFunc(graph, neighbor, visited)
            
    return visited


# ==========================================
# [BFS] 너비 우선 탐색 (큐 Queue 사용)
# ==========================================
def bfsFunc(graph, start):
    visited = [start]            # 방문한 노드를 담을 리스트
    queue = deque([start])  # 탐색할 노드를 담아둘 큐 (시작 노드 삽입)
    
    # 큐가 빌 때까지 반복
    while queue:
        # 1. 큐에서 가장 먼저 들어온 노드를 뽑음
        node = queue.popleft()
        
        # 2. 아직 방문하지 않은 노드라면
        if node not in visited:
            visited.append(node) # 방문 처리
            
            # 3. 현재 노드와 연결된 이웃들을 큐에 모두 대기시킴
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    
    return visited

# ==========================================
# 🚀 실행 및 결과 확인
# ==========================================
print("DFS 방문 순서:", dfsFunc(graph, 'A'))
# 예상 출력: A -> B -> D -> E -> F -> C

print("BFS 방문 순서:", bfsFunc(graph, 'A'))
# 예상 출력: A -> B -> C -> D -> E -> F