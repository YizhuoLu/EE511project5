import numpy as np
import matplotlib.pyplot as plt

# part 1
def Scwefel(x, y):
    return (418.9829 * 2 - X1 * np.sin(np.sqrt(abs(X1))) - X2 * np.sin(np.sqrt(abs(X2))))
x1 = np.linspace(-500, 500, 1000)
x2 = np.linspace(-500, 500, 1000)
X1, X2 = np.meshgrid(x1, x2)
plt.figure(1)
plt.title('Contour plot of the surface for the 2-D surface')
plt.xlabel('X1')
plt.ylabel('X2')
plt.contour(X1, X2, Scwefel(X1, X2))


# part 2
def ScwFunc(x):
    if sum(abs(x)>=500) > 0:
        return float('Inf')
    # check the dimension
    n = np.size(x)
    if n <= 1:
        return 418.9829 - x * np.sin(np.sqrt(abs(x)))
    else:
        return 418.9829 * n - sum(map(lambda m: m * np.sin(np.sqrt(abs(m))), x))

def simuAnneal(iters):
    k = 100
    X_samples = []
    Y_samples = []
    t = 1
    a = 0.99
    for i in range(k):
        K0 = 500
        H = K0
        a1 = np.array([1000 * (np.random.rand() - 0.5), 1000 * (np.random.rand() - 0.5)])
        for j in range(iters):
            H = H * a
            a2 = a1 + np.random.normal(0, 100, 2)
            alpha = np.exp((ScwFunc(a1) - ScwFunc(a2))/(H))
            alpha = min(1, alpha)
            if np.random.rand() < alpha:
                a1 = a2
        X_samples.append(a1)
        Y_samples.append(ScwFunc(a1))
    return X_samples, Y_samples

Min_X, Min_Value = simuAnneal(1000)
x_min = np.argmin(Min_Value)
print("The global minimum of this surface is:", Min_Value[x_min])
print("The global minimum point of this surface is:", Min_X[x_min])

# part 3
plt.figure(2)
plt.subplot(221)
poly_Min_X1, poly_Min_Value1 = simuAnneal(20)
plt.title("Histogram of polynomial t = 20")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(poly_Min_Value1, edgecolor='r')

plt.subplot(222)
poly_Min_X2, poly_Min_Value2 = simuAnneal(50)
plt.title("Histogram of polynomial t = 50")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(poly_Min_Value2, edgecolor='r')

plt.subplot(223)
poly_Min_X3, poly_Min_Value3 = simuAnneal(100)
plt.title("Histogram of polynomial t = 100")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(poly_Min_Value3, edgecolor='r')

plt.subplot(224)
poly_Min_X4, poly_Min_Value4 = simuAnneal(1000)
plt.title("Histogram of polynomial t = 1000")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(poly_Min_Value4, edgecolor='r')
# exponential
def ExpAnneal(iters):
    k = 100
    path = []
    X_samples = []
    Y_samples = []
    t = 1
    a = 0.5
    for i in range(k):
        path1 = []
        K0 = 500
        a1 = np.array([-400, -400])
        path1.append(a1)
        for j in range(iters):
            H = K0 * np.exp(-a * j)
            a2 = a1 + np.random.normal(0, 100, 2)
            alpha = np.exp((ScwFunc(a1) - ScwFunc(a2))/(H))
            alpha = min(1, alpha)
            if np.random.rand() < alpha:
                a1 = a2
                path1.append(a1)
        path.append(path1)
        X_samples.append(a1)
        Y_samples.append(ScwFunc(a1))
    return X_samples, Y_samples, path

plt.figure(3)
plt.subplot(221)
exp_Min_X1, exp_Min_Value1, path1 = ExpAnneal(20)
plt.title("Histogram of exponential t = 20")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(exp_Min_Value1, edgecolor='r')

plt.subplot(222)
exp_Min_X2, exp_Min_Value2, path2 = ExpAnneal(50)
plt.title("Histogram of exponential t = 50")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(exp_Min_Value2, edgecolor='r')

plt.subplot(223)
exp_Min_X3, exp_Min_Value3, path3 = ExpAnneal(100)
plt.title("Histogram of exponential t = 100")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(exp_Min_Value3, edgecolor='r')

plt.subplot(224)
exp_Min_X4, exp_Min_Value4, path4 = ExpAnneal(1000)
plt.title("Histogram of exponential t = 1000")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(exp_Min_Value4, edgecolor='r')
# logarithmic
def LogAnneal(iters):
    k = 100
    X_samples = []
    Y_samples = []
    t = 1
    a = 1000
    for i in range(k):
        K0 = 500 * np.log(2)
        a1 = np.array([1000 * (np.random.rand() - 0.5), 1000 * (np.random.rand() - 0.5)])
        for j in range(iters):
            H = K0 / np.log(a + j)
            a2 = a1 + np.random.normal(0, 100, 2)
            alpha = np.exp((ScwFunc(a1) - ScwFunc(a2)) / (H))
            alpha = min(1, alpha)
            if np.random.rand() < alpha:
                a1 = a2
        X_samples.append(a1)
        Y_samples.append(ScwFunc(a1))
    return X_samples, Y_samples

plt.figure(4)
plt.subplot(221)
log_Min_X1, log_Min_Value1 = LogAnneal(20)
plt.title("Histogram of logarithmic t = 20")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(log_Min_Value1, edgecolor='r')

plt.subplot(222)
log_Min_X2, log_Min_Value2 = LogAnneal(50)
plt.title("Histogram of logarithmic t = 50")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(log_Min_Value2, edgecolor='r')

plt.subplot(223)
log_Min_X3, log_Min_Value3 = LogAnneal(100)
plt.title("Histogram of logarithmic t = 100")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(log_Min_Value3, edgecolor='r')

plt.subplot(224)
log_Min_X4, log_Min_Value4 = LogAnneal(1000)
plt.title("Histogram of logarithmic t = 1000")
plt.xlabel('X')
plt.ylabel('Y')
plt.hist(log_Min_Value4, edgecolor='r')

# part 4
x4, v4, pathh = ExpAnneal(1000)
MinIndex = np.argmin(v4)
points = np.array(pathh[MinIndex]).T
plt.figure(5)
plt.contour(X1, X2, Scwefel(X1, X2))
plt.hold(True)
plt.plot(points[0], points[1], '*-', color='b')
plt.title("2-D sample path on the contour plot")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()