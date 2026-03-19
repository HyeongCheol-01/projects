import sys
N, K = map(int, sys.stdin.readline().split())
a = []

for i in range(1, N+1):
    if N % i ==0:
        a.append(i)
try:
    print(a[K-1])
except IndexError:
    print("0")