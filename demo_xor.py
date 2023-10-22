import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activation_functions import tanh
from loss import mse, mse_prime
from network import train, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    tanh(),
    Dense(3, 1),
    tanh()
]

# train
train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

# decision boundary plot
points = []
z_matrix = np.zeros((20, 20))
for i, x in enumerate(np.linspace(0, 1, 20)):
    for j, y in enumerate(np.linspace(0, 1, 20)):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0, 0]])
        z_matrix[i, j] = z[0, 0]

points = np.array(points)

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
ax.set_title("3D Scatter Plot")

# 2D contour plot
ax2 = fig.add_subplot(122)
X_grid, Y_grid = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
contour = ax2.contourf(X_grid, Y_grid, z_matrix.T, levels=20, cmap="winter")
ax2.set_title("Contour Plot")
plt.colorbar(contour)

plt.tight_layout()
plt.show()