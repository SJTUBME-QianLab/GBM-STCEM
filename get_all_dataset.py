import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def TDAS_f(shape,theta, beta):
    x = np.linspace(0, 2 * theta*np.pi, shape[0])
    y = np.linspace(-beta, beta, shape[1])
    z = np.linspace(0, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    V_data = np.sin(X) + np.tanh(Y)
    z_slice_index = shape[2] // 2
    V_data_slice = V_data[:, :, z_slice_index]
    return V_data_slice, X, Y, Z, z_slice_index



if __name__ == '__main__':
    shape = (50, 50, 50)
    theta = 1
    beta = 3
    V_data_slice, X, Y, Z, z_slice_index = TDAS_f(shape, theta, beta)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X[:, :, z_slice_index], Y[:, :, z_slice_index], V_data_slice,
                           cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("V_data (surface)")
    ax.set_title("3D Surface of V_data at Fixed Z")

    plt.show()