import sys

n = 0
a = []
while n==-1:
    n= int(sys.stdin.readline())
    a = []
    for i in range(1, n+1):
        if n % i ==0:
            a.append(i-1)







