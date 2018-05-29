import numpy as np
import matplotlib.pyplot as plt
import time
import math

def cityLocation(N):
    x = np.random.uniform(0, 1000, N)
    y = np.random.uniform(0, 1000, N)
    return x, y

# swap the city
def CitySwap(path):
    UpdatedPath = path[:]
    m = np.random.randint(0, len(path))
    n = np.random.randint(0, len(path))
    temp = UpdatedPath[m]
    UpdatedPath[m] = UpdatedPath[n]
    UpdatedPath[n] = temp
    return UpdatedPath

# start with a random path
path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
newPath = CitySwap(path)
print(path)
print(newPath)

# calculate the Euclidean distance
def getDistance(N1, N2, x, y):
    x1 = x[N1]
    y1 = y[N1]
    x2 = x[N2]
    y2 = y[N2]
    return math.sqrt(((x1-x2)**2 + (y1-y2)**2))

print(time.time())

def totalLength(path, x, y):
    total = 0
    for i in range(1, len(path)):
        total += getDistance(path[i], path[i-1], x, y)
    return total

# begin the iterated process
def runingMan(N, T0, a, MAXTIMES):
    t = T0
    a = a
    # obtain the city locations
    x, y = cityLocation(N)
    # start time
    time0 = time.time()
    note = []
    path_pre = []
    for i in range(N):
        path_pre.append(i)
    pace = []
    g = 0
    while(g < MAXTIMES):
        preLen = totalLength(path_pre, x, y)
        note.append(preLen)
        pace.append(g)
        path_now = CitySwap(path_pre)
        nowLen = totalLength(path_now, x, y)
        accept_rate = 0
        if preLen > nowLen:
            accept_rate = 1
        else:
            accept_rate = math.exp((preLen - nowLen)/t)
        u = np.random.uniform(0, 1)
        if u < accept_rate:
            path_pre = path_now
        else:
            path_pre = path_pre
        g += 1
        t = t * a
    time1 = time.time()
    print("The time that the whole process spend is:", format(time1 - time0))
    plt.title("The total length curve after iterations")
    plt.xlabel("# of iterations")
    plt.ylabel("path length")
    plt.plot(pace, note)
    plt.show()
    return x, y, path_pre

x, y, path = runingMan(1000, 1000, 0.99, 10000)

print(path)
x_route = []
y_route = []
for i in range(len(path)):
    x_route.append(x[path[i]])
    y_route.append(y[path[i]])
    plt.scatter(x[i], y[i], color='r')
    plt.hold = True
plt.title("The optimal path")
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_route, y_route, '*-', color='b')
print("current length is:", totalLength(path, x, y))
plt.show()