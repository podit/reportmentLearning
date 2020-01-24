import matplotlib.pyplot as plt
import numpy as np

l = 1000

x = np.zeros(l)
y = np.zeros(l)
e = 0.5

d = 1.5

ed = l // d

edr = e / ed

for i in range(l):
    x[i] = i
    e -= edr
    y[i] = e

print(x, y)

plt.plot(x, y)
plt.show()
