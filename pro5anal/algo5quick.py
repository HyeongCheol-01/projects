# 리스트 안에 들어 있는 자료를 오름차순 정렬
# 4) Quick 정렬
# 하나의 기준점을 중심으로 작은값과 큰값을 나눠서 각각 정렬 후
# 마지막에 이어 붙이는 방법
# g1 : 기준값 보다 작은 그룹
# 기준값
# g2 : 기준값 보다 큰 그룹

# 방법1 : 이해 위주
def quick_sort(a):
    n = len(a)
    if n <= 1:
        return a
    
    # 기준값 (편의상 가장 마지막 값을 취함
    pivot = a[-1]
    
    g1 = []     # 기준값보다 작은 그룹
    g2 = []     # 기준값보다 큰 그룹
    
    for i in range(0, n-1):
        if a[i] < pivot:
            g1.append(a[i])
        else:
            g2.append(a[i])
            
    print('g1 : ', g1)
    print('g2 : ', g2)
    
    return quick_sort(g1) + [pivot] + quick_sort(g2)
    
d = [6,8,3,1,2,4,7,5]
print(quick_sort(d))

print()
# 방법2 : 일반 알고리즘
def quick_sort2_sub(a, start, end):
    # 종료 조건: 정렬 대상이 한 개 이하일 때 정렬 X
    if end - start <= 0:
        return
    
    pivot = a[end]
    i = start
    for j in range(start, end):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            print(f"{a[i]}, {a[j]}, = {a[j]}, {a[i]}")
            i += 1
            
    a[i], a[end] = a[end], a[i]
    print(f"{a[i]}, {a[end]}, = {a[end]}, {a[i]}")
        
    quick_sort2_sub(a,start,i-1)    # 기준값보다 작은 그룹 재귀로 다시 정렬
    quick_sort2_sub(a,i+1,end)    # 기준값보다 큰 그룹 재귀로 다시 정렬
    
    
def quick_sort2(a):
    quick_sort2_sub(a, 0, len(a) - 1)   # 자료, 시작인덱스, 끝인덱스
    
    
d = [6,8,3,1,2,4,7,5]
quick_sort2(d)
print(d)