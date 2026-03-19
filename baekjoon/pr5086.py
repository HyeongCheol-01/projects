import sys
a=1
b=1

try:
    while a != 0 and b!= 0:
        a, b = map(int, sys.stdin.readline().split())
        if a % b == 0:
            print("multiple")
        elif b % a == 0:
            print("factor")
        else:
            print('neither')
except: ZeroDivisionError