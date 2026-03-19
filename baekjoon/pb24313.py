import sys

a,b=map(int, sys.stdin.readline().split())
c=int(input())
n=int(input())


if a*n+b <= c*n and a<=c :
    print(1)
else:
    print(0)
