import numpy as np


import taichi as ti
ti.init(arch=ti.gpu)  # Alternatively, ti.init(arch=ti.cpu)

nx, ny = 101, 101
f = ti.field(dtype=ti.f32, shape=(nx+1, ny+1, 10))
resf = ti.field(dtype=ti.f32, shape=(nx+1, ny+1, 10))
feq = ti.field(dtype=ti.f32, shape=(nx+1, ny+1, 10))
u = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))  # --> u
v = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))
rho = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))

w_tuple = (-22222223, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9)
w = ti.field(dtype=ti.f32, shape=10)
cx_tuple = (-22222223, 1, 0, -1, 0, 1, -1, -1, 1, 0)
cy_tuple = (-22222223, 0, 1, 0, -1, 1, 1, -1, -1, 0)
cx = ti.field(dtype=ti.i32, shape=10)  # ti.i32 ?
cy = ti.field(dtype=ti.i32, shape=10)
invert_k_tuple = (-22222223, 3, 4, 1, 2, 7, 8, 5, 6, 9)
invert_k = ti.field(dtype=ti.i32, shape=10)

uo = 0.10
alpha = 0.01
omega = 1.0 / (3.0 * alpha + 0.5)
count = 0

for k in range(1, 10):
    w[k] = w_tuple[k]
    cx[k] = cx_tuple[k]
    cy[k] = cy_tuple[k]
    invert_k[k] = invert_k_tuple[k]

for i in range(1, nx +1):  # 1, 2 ... nx
    u[i, ny] = uo
@ti.kernel
def ti_init():
    for i, j in rho:
        rho[i, j] = 1


@ti.kernel
def boundary():
    for j in range(1, ny +1):  # 1, 2 ... ny
        f[1, j, 1] = f[1, j, 3]
        f[1, j, 5] = f[1, j, 7]
        f[1, j, 8] = f[1, j, 6]

        f[nx, j, 3] = f[nx, j, 1]
        f[nx, j, 7] = f[nx, j, 5]
        f[nx, j, 6] = f[nx, j, 8] #taichi?

    for i in range(1, nx +1):  # 1, 2 ... nx
        f[i, 1, 2] = f[i, 1, 4]
        f[i, 1, 5] = f[i, 1, 7]
        f[i, 1, 6] = f[i, 1, 8]

    for i in range(2, nx-1 +1):  # 2, 3 ... nx-1
        rhon = f[i, ny, 9] + f[i, ny, 1] + f[i, ny, 3] + 2.0 * (f[i, ny, 2] + f[i,ny,6] + f[i,ny,5])
        f[i, ny, 4] = f[i, ny, 2]
        f[i, ny, 8] = f[i, ny, 6] + rhon * uo / 6.0
        f[i, ny, 7] = f[i, ny, 5] - rhon * uo / 6.0


@ti.kernel
def collision():
    for j in range(1, ny +1):  # 1, 2 ... ny
        for i in range(1, nx +1):  # 1, 2 ... nx
            t1 = u[i, j] * u[i, j] + v[i, j] * v[i, j]
            for k in range(1, 10):  # 1, 2 ... 9
                t2 = u[i,j] * cx[k] + v[i,j] * cy[k]
                feq[i, j, k] = rho[i,j] * w[k] * (1.0 + 3.0 * t2 + 4.5 * t2 * t2 - 1.5 * t1)
                f[i, j, k] = (1.0 - omega) * f[i, j, k] + omega * feq[i, j, k]


@ti.kernel
def ruv():
    for i in range(1, nx +1):
        for j in range(1, ny +1):
            rho[i, j] = 0.0
            for k in range(1, 10):
                rho[i, j] += f[i, j, k]

    for i in range(1, nx +1):  # 1, 2 ... nx
        rho[i, ny] = f[i, ny, 9] + f[i, ny, 1] + f[i, ny, 3] + 2.0 * (f[i, ny, 2] + f[i, ny, 6] + f[i, ny, 5])

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            u[i, j] = f[i, j, 1] + f[i, j, 5] + f[i, j, 8] - ( f[i, j, 3] + f[i, j, 6] + f[i, j, 7] )
            u[i, j] /= rho[i, j]

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            v[i, j] = f[i, j, 2] + f[i, j, 5] + f[i, j, 6] - ( f[i, j, 4] + f[i, j, 7] + f[i, j, 8] )
            v[i, j] /= rho[i, j]

@ti.func
def copy_resf_to_f():
    for a, b, c in f:
        f[a, b, c] = resf[a, b, c]
@ti.kernel
def stream():
    for i in range(1, nx +1):
        for j in range(1, ny +1):
            for k in range(1, 9):  # 1, 2 ... 8
                k_inv = invert_k[k]
                ii = i + cx[k_inv]
                jj = j + cy[k_inv]
                # if ii < 1 or ii > nx or jj < 1 or jj > ny:
                #     continue

                if ii == 0:
                    ii = nx
                if jj == 0:
                    jj = ny
                if ii == nx + 1:
                    ii = 1
                if jj == ny + 1:
                    jj = 1

                resf[i, j, k] = f[ii, jj, k]

            resf[i, j, 9] = f[i, j, 9]

    copy_resf_to_f()


def print_rho(count, comment):
    print(f'count {count}, {comment}')
    # for i in range(1, 12):
    #     for j in range(1, 12):
    #         print(f'{rho[i, j]:9f}', end=' ')
    #     print()

    print(f'{ny}  ', end='')
    rho_here = 223.0
    for i in range(1, nx +1):  # 1, 2 ... nx
        rho_here = f[i, ny, 9] + f[i, ny, 1] + f[i, ny, 3] + 2.0 * (f[i, ny, 2] + f[i, ny, 6] + f[i, ny, 5])
        print(f'{rho_here:9f} ', end='')
    print()

    for j in reversed(range(1, ny-1 +1)):  # ny-1, ny-2 ... 2, 1
        print(f'{j}  ', end='')
        for i in range(1, nx +1):
            rho_here = 0.0
            for k in range(1, 10):
                rho_here += f[i, j, k]
            print(f'{rho_here:9f} ', end='')
        print()


    print()

def print_f(count, comment):
    print(f'count {count}, {comment}. print_f')
    for k in range(1, 10):
        print(f'\nval(:,:,{k}) =\n')
        for i in range(1, 12):
            # print(f'i{i:<3d}', end=' ')
            for j in range(1, 12):
                print(f'    {f[i,j,k]:6.4f}', end='')
            print()
    print()

def print_rho_in_ruv(count, comment):
    print(f'count {count}, {comment}')
    for i in range(1, 12):
        for j in range(1, 12):
            print(f'{rho[i, j]:9f}', end=' ')
        print()
    print()

@ti.kernel
def tmp_first_collision():
    for j in range(1, ny + 1):  # 1, 2 ... ny
        for i in range(1, nx + 1):  # 1, 2 ... nx
            t1 = u[i, j] * u[i, j] + v[i, j] * v[i, j]
            for k in range(1, 10):  # 1, 2 ... 9
                t2 = u[i, j] * cx[k] + v[i, j] * cy[k]
                feq[i, j, k] = rho[i, j] * w[k] * (1.0 + 3.0 * t2 + 4.5 * t2 * t2 - 1.5 * t1)
                f[i, j, k] = (1.0 - omega) * f[i, j, k] + omega * feq[i, j, k]

# main loop
ti_init()
# tmp_first_collision()
while count < 1000:
    # print_rho(count, "loop begin not from ruv")
    # print_rho_in_ruv(count, "loop begin from ruv")

    # print_f(count, "loop begin")

    collision()
    # print_f(count, "after collision")

    stream()
    # print_f(count, "after stream")

    boundary()
    # print_f(count, "after boundary")

    ruv()
    # print_f(count, "after ruv")
    # print_rho(count, "after ruv, not from ruv")
    # print_rho_in_ruv(count, "after ruv, from ruv")

    count += 1


# print_f(count, "whole loop over")
def save_fruv():
    np_f = np.zeros((nx, ny, 9))
    np_rho = np.zeros((nx, ny))
    np_u = np.zeros((nx, ny))
    np_v = np.zeros((nx, ny))

    for i in range(1, nx +1):
        for j in range(1, ny +1):
            np_rho[i-1, j-1] = rho[i, j]
            np_u[i-1, j-1] = u[i, j]
            np_v[i-1, j-1] = v[i, j]

            for k in range(1, 10):
                np_f[i-1, j-1, k-1] = f[i, j, k]

    np.save("np_f.npy", np_f)
    np.save("np_rho.npy", np_rho)
    np.save("np_u.npy", np_u)
    np.save("np_v.npy", np_v)

save_fruv()

