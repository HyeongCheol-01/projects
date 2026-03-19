import sys

a, b, c= map(int, sys.stdin.readline().split())
d = sorted([a,b,c])



if d[0] + d[1] <= d[2]:
    print(2*(d[0] + d[1])-1)

else:
    print(d[0] + d[1] + d[2])