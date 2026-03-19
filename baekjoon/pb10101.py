a = []
hap = 0
for _ in range(3):
    b = int(input())
    hap += b
    a.append(b)
if hap !=180:
    print('Error')
elif a[0]==a[1]==a[2]:
    print('Equilateral')
elif a[0]!=a[2]!=a[1]:
    print('Scalene')
else:
    print('Isosceles')