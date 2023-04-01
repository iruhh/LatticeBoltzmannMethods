import taichi as ti
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu)

nx, ny = 101, 101
f = ti.field(dtype=ti.f32, shape=(nx+1, ny+1, 10))
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

uo = 0.10
alpha = 0.01
omega = 1.0 / (3.0 * alpha + 0.5)
count = 0

for k in range(1, 10):
    w[k] = w_tuple[k]
    cx[k] = cx_tuple[k]
    cy[k] = cy_tuple[k]

for i in range(1, nx + 1):  # 1, 2 ... nx
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
    for i, j in rho:
        rho[i, j] = 0.0

    for i, j, k in f:
        if not (k == 0):
            rho[i, j] += f[i, j, k]

    for i in range(1, nx +1):  # 1, 2 ... nx
        rho[i, ny] = f[i, ny, 9] + f[i, ny, 1] + f[i, ny, 3] + 2.0 * (f[i, ny, 2] + f[i, ny, 6] + f[i, ny, 5])

    for i, j in u:
        if i == 0 or j == 0:
            continue

        u[i, j] = f[i, j, 1] + f[i, j, 5] + f[i, j, 8] - ( f[i, j, 3] + f[i, j, 6] + f[i, j, 7] )
        u[i, j] /= rho[i, j]


    for i, j in v:
        if i == 0 or j == 0:
            continue

        v[i, j] = f[i, j, 2] + f[i, j, 5] + f[i, j, 6] - ( f[i, j, 4] + f[i, j, 7] + f[i, j, 8] )
        v[i, j] /= rho[i, j]


@ti.kernel
def stream():
    for i in range(1, nx +1):
        for j in range(1, ny +1):
            for k in range(1, 9):  # 1, 2 ... 8
                ii = i + cx[k]
                jj = j + cy[k]
                if ii < 1 or ii > nx or jj < 1 or jj > ny:
                    continue
                f[i, j, k] = f[ii, jj, k]


def print_rho(count):
    print(count)
    for i in range(1, 11):
        for j in range(1, 8):
            print(rho[i, j], end=' ')
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

while count < 100:
    collision()
    stream()
    boundary()
    ruv()

    count += 1

    if count % 10 == 0:
        print_rho(count)





