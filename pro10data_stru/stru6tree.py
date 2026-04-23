# Tree


# 회사 조직도 구조 Tree로 표현

company = {
    'CEO':['개발팀장','기획팀장','영업팀장'],
    '개발팀장':['백엔드','프론트엔드'],
    '기획팀장':['서비스기획'],
    '영업팀장':['국내영업','해외영업'],
    '백엔드':[],
    '프론트엔드':[],
    '서비스기획':[],
    '국내영업':[],
    '해외영업':[]
}

# 트리 구조 함수
def showTree(node, level):
    print(' ' * level + ' - ' + node)
    # 현재 노드의 자식 출력
    for child in company[node]:
        showTree(child, level + 1)  # 재귀
        
print('회사조직도')
showTree('CEO',0)
    
    
    