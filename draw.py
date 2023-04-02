import numpy as np
import matplotlib.pyplot as plt

f = np.load("np_f.npy")
rho = np.load("np_rho.npy")
u = np.load("np_u.npy")
v = np.load("np_v.npy")
np.save(".\\data\\f.npy", f)
np.save(".\\data\\rho.npy", rho)
np.save(".\\data\\u.npy", u)
np.save(".\\data\\v.npy", v)
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
    startx = np.linspace(0, 1.0, nx)
    starty = np.linspace(0, 1.0, ny)
    start_points = np.stack([startx, starty], axis=-1)

    fig = plt.figure(figsize=(12, 7))
    plt.streamplot(X, Y, v, u)  # why here is v_fig, u_fig, but not u_fig, v_fig

    str = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(1, ny):
            str[i, j] = str[i, j-1] + 0.25 * (rho[i, j] + rho[i, j-1]) * (u[i, j] + u[i, j-1])

    # plt.contour(X, Y, str, 80, cmap='RdBu_r')
    plt.show()


draw_figures()