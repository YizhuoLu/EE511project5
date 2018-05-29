import numpy as np
import scipy.stats as st
import math
import matplotlib.pyplot as plt

def pi(x):
    return 0.6 * st.beta.pdf(x, 1, 8) + 0.4 * st.beta.pdf(x, 9, 1)

def q(x, x_now, var):
    return (1/(math.sqrt(2*math.pi)*var)) * math.exp(-(x - x_now)**2/(2*var**2))

iters = 5000
def metropolis_hastings(x_0, var):
    samples = []
    samples.append(x_0)
    for i in range(iters):
        x_cand = np.random.normal(x_0, var)
        alpha = min(1, (q(x_0, x_cand, var)*pi(x_cand))/(q(x_cand, x_0, var)*pi(x_0)))
        if np.random.rand() <= alpha:
            x_0 = x_cand
            samples.append(x_0)
        else:
            samples.append(x_0)
    return samples

plt.figure(1)
plt.subplot(411)
plt.title("sample paths with different initial points")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.1, 0.1))
plt.subplot(412)
plt.title("'# of experiments")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.2, 0.1))
plt.subplot(413)
plt.title("'# of experiments")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.4, 0.1))
plt.subplot(414)
plt.title("'# of experiments")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.8, 0.1))

plt.figure(2)
plt.subplot(411)
plt.title("Sample Paths with different variance")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.2, 0.05))
plt.subplot(412)
plt.title("'# of experiments")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.2, 0.1))
plt.subplot(413)
plt.title("'# of experiments")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.2, 0.2))
plt.subplot(414)
plt.title("'# of experiments")
plt.xlabel('# of experiments')
plt.ylabel('samples')
plt.plot(range(iters+1), metropolis_hastings(0.2, 0.4))
plt.figure(3)
plt.title("histogram of samples using Metropolis-Hasting algorithm")
plt.xlabel('X')
plt.ylabel('Y')
m = np.arange(0, 1, 0.02)
plt.hist(metropolis_hastings(0.1, 0.1), bins=m, histtype='bar', edgecolor='r')
plt.show()