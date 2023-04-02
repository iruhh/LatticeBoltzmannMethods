import numpy as np
import matplotlib.pyplot as plt

f = np.load("np_f.npy")
rho = np.load("np_rho.npy")
u = np.load("np_u.npy")
v = np.load("np_v.npy")
np.save("LidDriven2_np_f.npy", f)
np.save("idDriven2_np_rho.npy", rho)
np.save("idDriven2_np_u.npy", u)
np.save("idDriven2_np_v.npy", v)
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
    startx = np.linspace(0, 1.0, nx * 10)
    starty = np.linspace(0, 1.0, ny * 10)
    start_points = np.stack([startx, starty], axis=-1)

    fig = plt.figure(figsize=(12, 7))
    # plt.streamplot(X_fig, Y_fig, v_fig, u_fig)  # why here is v_fig, u_fig, but not u_fig, v_fig

    str = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(1, ny):
            str[i, j] = str[i, j-1] + 0.25 * (rho[i, j] + rho[i, j-1]) * (u[i, j] + u[i, j-1])

    plt.contour(X, Y, str, 80, cmap='RdBu_r')
    plt.show()


draw_figures()