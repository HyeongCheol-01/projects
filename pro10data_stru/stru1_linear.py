# LinearLit(선형구조) 연습
# 놀이공원(공연장) 줄서기

# 연습1 - python 함수 사용
line = ['철수', '영희', '민수']
print('현재 줄 상태 : ', line)
print()

# 데이터 접근 - 인덱스를 사용(빠름)
print('맨 앞 사람:', line[0])
print('두 번쨰 사람:',line[1])
print()

#새치기(삽입)
line.insert(2, '지수') # 민수는 다음자리로 밀림
print('지수 삽입 후 현재 줄 상태 :', line)
print()

# 줄에서 빠지기(삭제)
line.remove('영희') # 1번쨰 이후 자료가 앞으로 한칸씩 이동
print('영희가 빠진후 줄 상태:', line)
print()

# 앞사람 부터 놀이기구 타기- 첫번째 자료부터 빠져나감. 뒤자료는 앞으로 이동
first_person = line.pop(0)  # pop(0): 왼쪽값 추출, pop() 오른쪽값 추출
print('첫번째 사람 입장 후 남은 줄 상태:',line)
print()

# 현재 남은 사람 변화와 함께 출력
for i, p in enumerate(line):
    print(i, '번째 사람:', p)
print('*'*20)

# 연습2 - python 코드를 사용
line = ['철수', '영희', '민수']
print('현재 줄 상태 : ', line)
print()

# 데이터 접근 - 인덱스를 사용(빠름)
print('맨 앞 사람:', line[0])
print('두 번쨰 사람:',line[1])
print()

# 중간에 새로운 사람이 삽입 : ' 철수','영희','민수' 에서 지수를 민수 앞에 끼워넣기
# index 2 위치에 지수 삽입 - 공간확보 -> index 2 이후 뒤로 한 칸씩 이동 -> 값 대입
line.append(None) #공간확보
for i in range(len(line) -1, 2, -1):
    line[i] = line[i-1]
print(line)
line[2] = '지수'
print('지수가 삽입된 후 줄 상태', line)
print()


# 중간에 사람 삭제
remove_index = None
for i in range(len(line)):
    if line[i] == '영희':
        remove_index = i
        break
    
# 앞으로 한 칸씩 이동
for i in range(remove_index, len(line)-1):
    line [i] = line[i+1]
print(line)
line.pop()
print('영희가 빠져 나간 후 줄 상태 :', line )

# 앞사람 부터 놀이기구 타기
# 앞에서 부터 한 칸 씩 좌측으로 이동
first_person = line[0]
for i in range(0, len(line)-1):
    line[i] = line[i+1]
    
print(line)
line.pop()
print('남아 있는 줄 상태:', line) 

# 현재 남은 사람 번호와 함께 출력
for i, p in enumerate(line):
    print(i, '번째 사람', p)
    
# LinearList는 index로 즉시 접근 가능, 삽입/삭제 시 데이터 이동 발생(비용) - 비효율적 