import sys

x,y,w,h = map(int, sys.stdin.readline().split())

a = min(x-0, w-x)
b = min(y-0, h-y)

print(min(a,b))