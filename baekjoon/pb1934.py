import sys

def get_gcd(a, b):
    while b != 0:
        a,b = b, a % b
    return a


n = int(input())
res=[]
for _ in range(n):
    a, b = map(int, sys.stdin.readline().split())
    lcm = (a*b)//get_gcd(a, b)
    res.append(lcm)
for i in range(n):
    print(res[i])
    
    