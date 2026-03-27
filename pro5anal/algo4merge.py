# 리스트 안에 들어 있는 자료를 오름차순 정렬
# 1) 병합(merge) 정렬
# 리스트 자료를 반으로 나눔. 요소가 1개씩 남을 때 까지 반복
# 분할된 리스트를 정렬하며 하나로 합친다 (정렬 상태 유지)

# 방법1 : 이해 위주

def merge_sort(a):
    n = len(a)
    
    if n <= 1:
        return a
    
    mid = n // 2 # 중간을 기준으로 두 그룹으로 분할
    # 함수는 독립적인 공간을 갖음. 아래의 g1, g2는 서로 불간섭
    g1 = merge_sort(a[:mid])    # 재귀
    print('g1 : ', g1)
    g2 = merge_sort(a[mid:])
    print('g2 : ', g2)

    # 두 그룹
    result=[]    
    while g1 and g2:
        print(g1[0], ' ',g2[0])
        if g1[0] < g2[0]:
            result.append(g1.pop(0))
        else:
            result.append(g2.pop(0))
        print('result : ', result)
        
    # g1과 g2중 소진된 것은 스킵
    while g1:
        result.append(g1.pop(0))
        
    while g2:
        result.append(g2.pop(0))
    
    return result

d = [6,8,3,1,2,4,7,5]
print(merge_sort(d))

print()
# 방법2 : 일반 알고리즘
# 재귀호출이 정렬된 리스트를 반환.
# 병합도 새 리스트를 만들어 반환.
# 원본 리스트는 그대로이고 정렬된 결과는 새 리스트에 저장

def merge_sort2(a):
    if len(a) <= 1:
        return a
    
    mid = len(a) // 2
    left = merge_sort2(a[:mid])
    right = merge_sort2(a[mid:])
    
    result = []
    i = j = 0
    
    # 병합
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 남은 요소 추가
    result += left[i:]
    result += right[j:]
    return result
d = [6,8,3,1,2,4,7,5]
sorted_d = merge_sort2(d)
print(sorted_d)



