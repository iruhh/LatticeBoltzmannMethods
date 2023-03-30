import taichi as ti
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu)

nx, ny = 1000, 500
Q = 9
omega = 1.0
# Once a field is declared, Taichi automatically initializes its elements to zero.
isObstacle = ti.field(dtype=ti.i32, shape=(nx, ny))
fa = ti.field(dtype=ti.f32, shape=(nx, ny, Q))
fb = ti.field(dtype=ti.f32, shape=(nx, ny, Q))
# f6 f3 f5
# f2 f0 f1
# f8 f4 f7
w_tuple = (1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18)
ex_tuple = (0, 1, -1, 0, 0, 1,-1, 1, -1)
ey_tuple = (0, 0, 0, 1, -1, 1, 1, -1, -1)
invert_l_tuple = (0, 2, 1, 4, 3, 8, 7, 6, 8)
w = ti.field(dtype=ti.f32, shape=Q)
ex = ti.field(dtype=ti.i32, shape=Q)
ey = ti.field(dtype=ti.i32, shape=Q)
invert_l = ti.field(dtype=ti.i32, shape=Q)
for i in range(Q):
    w[i] = w_tuple[i]
    ex[i] = ex_tuple[i]
    ey[i] = ey_tuple[i]
    invert_l[i] = invert_l_tuple[i]


@ti.kernel
def initialize():
    for i, j in isObstacle:
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            isObstacle[i, j] = 1

    for i, j, l in fa:
        fa[i, j, l] = w[l]

    for i, j, l in fb:
        fb[i, j, l] = w[l]


@ti.kernel
def stream():
    for i, j, l in fb:
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            continue
        l_inv = invert_l[l]
        ii, jj = i + ex[l_inv], j + ey[l_inv]
        if ii < 0 or ii > nx-1 or jj < 0 or jj > ny-1:
            continue
        if isObstacle[ii, jj] == 1:
            fb[i, j, l] = fa[i, j, l_inv]
        else:
            fb[i, j, l] = fa[ii, jj, l]


@ti.func
def get_density_velocity(i, j):
    density, ux, uy = 0.0, 0.0, 0.0
    for l in ti.ndrange((0, 9)):
        fb_ijl = fb[i, j, l]
        density += fb_ijl
        ux += ex[l] * fb_ijl
        uy += ey[l] * fb_ijl
    return density, ux, uy

@ti.kernel
def collide():
    for i, j, l in fb:
        if isObstacle[i, j] == 1:
            continue

        density, ux, uy = get_density_velocity(i, j)

        for l in ti.ndrange((0, 9)):
            a = ex[l] * ux + ey[l] * uy
            f_eq = w[l] * (density + 1.5 * (ux * ux + uy * uy) + 3 * a + 4.5 * a * a)
            fb[i, j, l] = (1.0 - omega) * fb[i, j, l] + omega * f_eq


@ti.kernel
def fill_fa_with_fb():
    for i, j, l in fa:
        fa[i,j,l] = fb[i,j,l]

@ti.func
def simple_check_get_density_ux_uy(i, j, isb):
    density, ux, uy = 0.0, 0.0, 0.0
    for l in range(Q):
        f_ijl = fb[i, j, l]
        if isb == False:
            f_ijl = fa[i, j, l]
        density += f_ijl
        ux += ex[l] * f_ijl
        uy += ey[l] * f_ijl
    return density, ux, uy

@ti.kernel
def simple_check():
    for i in range(975, 985):
        density, ux, uy = simple_check_get_density_ux_uy(i, 250, False)
        print(density, ux, uy)

initialize()
simple_check()
while True:
    stream()
    collide()
    fill_fa_with_fb()
    simple_check()

    # simple_check()