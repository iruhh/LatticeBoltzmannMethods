import numpy as np
import matplotlib.pyplot as plt

uo = 0.02
f = np.load("np_f.npy")
rho = np.load("np_rho.npy")
u = np.load("np_u.npy")
v = np.load("np_v.npy")
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_f.npy", f)
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_rho.npy", rho)
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_u.npy", u)
np.save(".\\data\\SteadyPlanePoiseuilleFlow2_v.npy", v)
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
    print(f'u_uniformed.shape {u_uniformed.shape}')
    print(f'u[51, :].shape {u[51, :].shape}, y.shape {y.shape}')
    print(f'u[51, :] {u[51, :]}')
    plt.plot(u[51, :], y)


    # contour
    str = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(1, ny):
            str[i, j] = str[i, j-1] + 0.5 * (u[i, j] + u[i, j-1])

    # plt.contour(X, Y, str, 80, cmap='RdBu_r')
    plt.show()


draw_figures()