# import taichi as ti
# ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu)
#
# nx, ny = 7, 5
# Q = 9
# alpha = 0.01
# omega = 1.0 / (3.0 * alpha + 0.5)
# # v
# # ^
# # |  -> u
# v_wall = 0.0
# u_wall = 0.1
# # Once a field is declared, Taichi automatically initializes its elements to zero.
# isObstacle = ti.field(dtype=ti.i32, shape=(nx, ny))
# fa = ti.field(dtype=ti.f32, shape=(nx, ny, Q))
# fb = ti.field(dtype=ti.f32, shape=(nx, ny, Q))
# rho = ti.field(dtype=ti.f32, shape=(nx, ny))
# vec = ti.field(dtype=ti.f32, shape=(nx, ny, 2))
#
# w_tuple = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
# ex_tuple = (0, 1, 0, -1,  0, 1, -1, -1,  1)
# ey_tuple = (0, 0, 1,  0, -1, 1,  1, -1, -1)
# invert_l_tuple = (0, 3, 4, 1, 2, 7, 8, 5, 6) # f0_inv = f0?
# w = ti.field(dtype=ti.f32, shape=Q)
# ex = ti.field(dtype=ti.i32, shape=Q)
# ey = ti.field(dtype=ti.i32, shape=Q)
# invert_l = ti.field(dtype=ti.i32, shape=Q)
#
# for i in range(Q):
#     w[i] = w_tuple[i]
#     ex[i] = ex_tuple[i]
#     ey[i] = ey_tuple[i]
#     invert_l[i] = invert_l_tuple[i]
#
#
# @ti.kernel
# def initialize():
#     for i, j in isObstacle:
#         if i == 0 or i == nx-1 or j == 0 or j == ny-1: # todo
#             isObstacle[i, j] = 1
#
#     for i, j, l in fa:
#         fa[i, j, l] = w[l]  # why?
#
#     for i, j, l in fb:
#         fb[i, j, l] = w[l]
#
#     for i, j in rho:
#         rho[i, j] = 1.0
#
#
# @ti.kernel
# def stream():
#     for i, j, l in fb:
#         if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
#             continue
#
#         l_inv = invert_l[l]
#         ii, jj = i + ex[l_inv], j + ey[l_inv]
#         if isObstacle[ii, jj] == 1:
#             fb[i, j, l] = fa[i, j, l_inv]
#         else:
#             fb[i, j, l] = fa[ii, jj, l]
#
#
#     # for i in range(2, nx-2): # 2, 3 ... nx-3
#     #     j = ny-2
#     #
#     #     # why density1 != density2?
#     #     # the point is: where did the fi towards wall go?
#     #
#     #     # density1 = 0.0
#     #     # for l in range(9):
#     #     #     density1 += fb[i,j,l]
#     #     # print(i, j, f'density1 {density1}')
#     #
#     #     # density3 = (fb[i, j, 0] + fb[i, j, 1] + fb[i, j, 3] + 2 * (fb[i, j, 2] + fb[i, j, 6] + fb[i, j, 5])) / (1.0 + v_wall)
#     #     # fb[i, j, 4] = fb[i, j, 2] - 2.0 / 3.0 * density3 * v_wall
#     #     # fb[i, j, 7] = fb[i, j, 5] + 0.5 * (fb[i, j, 1] - fb[i, j, 3]) - 1.0 / 6.0 * density3 * v_wall - 0.5 * density3 * u_wall
#     #     # fb[i, j, 8] = fb[i, j, 6] + 0.5 * (fb[i, j, 3] - fb[i, j, 1]) - 1.0 / 6.0 * density3 * v_wall + 0.5 * density3 * u_wall
#     #
#     #     density3 = fb[i, j, 0] + fb[i, j, 1] + fb[i, j, 3] + 2 * (fb[i, j, 2] + fb[i, j, 6] + fb[i, j, 5])
#     #     fb[i, j, 4] = fb[i, j, 2]
#     #     fb[i, j, 7] = fb[i, j, 5] - density3 * u_wall / 6.0
#     #     fb[i, j, 8] = fb[i, j, 6] + density3 * u_wall / 6.0
#     #
#     #
#     #     # density2 = 0.0
#     #     # for l in range(9):
#     #     #     density2 += fb[i, j, l]
#     #     # print(i, j, f'density2 {density2}')
#     #     #
#     #     # for l in range(9):
#     #     #     print(fb[i, j, l])
#     #     # print()
#
#
# @ti.func
# def update_density_velocity():
#     for i, j in rho:
#         rho[i, j] = 0.0
#
#     for i, j, l in fa:
#         rho[i, j] += fa[i, j, l]
#
#     for i in range(1, nx-1):  # 1, 2 ... nx-2
#         rho[i, ny-2] = fa(i, ny, 9) + fa(i, ny, 1) + fa(i, ny, 3) + 2 .* (fa(i, ny, 2) + fa(i, ny, 6) + fa(i, ny, 5));
#
#     # density, ux, uy = 0.0, 0.0, 0.0
#     # for l in range(9):
#     #     fb_ijl = fb[i, j, l]
#     #     density += fb_ijl
#     #
#     # if j == ny-2: # ny-2!!!!
#     #     density = fb[i, j, 0] + fb[i, j, 1] + fb[i, j, 3] + 2 * (fb[i, j, 2] + fb[i, j, 6] + fb[i, j, 5])
#     #
#     # for l in range(9):
#     #     fb_ijl = fb[i, j, l]
#     #     ux += ex[l] * fb_ijl
#     #     uy += ey[l] * fb_ijl
#     #
#     # return density, ux / density, uy / density
#
# @ti.kernel
# def collide():
#     for i, j, l in fb:
#         if isObstacle[i, j] == 1:
#             continue
#
#         density, ux, uy = get_density_velocity(i, j)
#         density = float(density)
#         ux = float(ux)
#         uy = float(uy)
#
#         a = float(ex[l]) * float(ux) + float(ey[l]) * float(uy)
#         f_eq = w[l] * density * (1.0 - 1.5 * (ux * ux + uy * uy) + 3.0 * a + 4.5 * a * a)
#         fb[i, j, l] = (1.0 - omega) * fb[i, j, l] + omega * f_eq
#
#
# @ti.kernel
# def fill_fa_with_fb():
#     for i, j, l in fa:
#         fa[i,j,l] = fb[i,j,l]
#
# initialize()
# collide() # ?
# fill_fa_with_fb()
# iu = 10000
# while iu > 0:
#     stream()
#     collide()
#     fill_fa_with_fb()
#
#     iu -= 1
