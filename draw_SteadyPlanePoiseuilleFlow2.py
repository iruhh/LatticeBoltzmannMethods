import numpy as np
import matplotlib.pyplot as plt

uo = 0.2
f = np.load("np_f.npy")
rho = np.load("np_rho.npy")
u = np.load("np_u.npy")
v = np.load("np_v.npy")
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_loop10000_f.npy", f)
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_loop10000_rho.npy", rho)
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_loop10000_u.npy", u)
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_loop10000_v.npy", v)
print(f.shape)
print(rho.shape)
print(u.shape)
print(v.shape)
nx = f.shape[0]
ny = f.shape[1]


def draw_figures():
    x = np.linspace(0, 1.0, nx)
    y = np.linspace(0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    X = np.transpose(X)
    Y = np.transpose(Y)


    # u_uniformed
    u_uniformed = u / uo  # to be confirmed
    for i in range(100, 701, 200):
        plt.plot(u_uniformed[i, :], y, label=f'x {i}')
        plt.legend()
    plt.show()

    # contour
    str = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(1, ny):
            str[i, j] = str[i, j-1] + 0.5 * (u[i, j] + u[i, j-1])

    plt.contour(X, Y, str, 80, cmap='RdBu_r')
    plt.show()


draw_figures()