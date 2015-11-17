import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 30)
y = np.cos(x)
plt.plot(x, y)
plt.xlim(-1, 5)

plt.show()
