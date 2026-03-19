import sys
e = []
count=0
while 1:
    a, b, c= map(int, sys.stdin.readline().split())
    d = sorted([a,b,c])

    if a == 0 and b==0 and c==0: break
    e.append(d)
    count += 1

for i in range(count):
    if e[i][0] + e[i][1] <= e[i][2]:
        print('Invalid')
    elif e[i][0]==e[i][1]==e[i][2]:
        print('Equilateral')
    elif e[i][0]!=e[i][1]!=e[i][2]:
        print('Scalene')
    else:
        print('Isosceles')


