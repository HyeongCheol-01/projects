import sys

N = int(sys.stdin.readline())
x_data=[]
y_data=[]
for _ in range(N):
    x, y = map(int, sys.stdin.readline().split())
    x_data.append(x)
    y_data.append(y)

x_min = min(x_data)
x_max = max(x_data)
y_min = min(y_data)
y_max = max(y_data)

square = ((x_max - x_min) * (y_max - y_min))
print(square)