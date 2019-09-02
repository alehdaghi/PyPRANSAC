from __future__ import print_function, absolute_import

from numba import vectorize, cuda, jit
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import numba as nb
import math


@cuda.jit(device=True, inline=True)
def cu_subtract_point(ax, ay, az, bx, by, bz):
    return ax - bx, ay - by, az - bz


@cuda.jit('float32(float32, float32, float32, float32, float32, float32)', device=True, inline=True)
def cu_dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@cuda.jit(device=True, inline=True)
def crossNormal(ax, ay, az, bx, by, bz):
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    s = math.sqrt(cx * cx + cy * cy + cz * cz)
    return cx / s, cy / s, cz / s


@cuda.jit(device=True, inline=True)
def cuRand(rng_states, a, b):
    thread_id = cuda.grid(1)
    x = xoroshiro128p_uniform_float32(rng_states, thread_id)
    return (nb.int32)(x * (b - a) + a)


@cuda.jit
def compute_pi(rng_states, iterations, out):
    """Find the maximum value in values and store in result[0]"""

    thread_id = cuda.grid(1)

    # print(cuda.grid(1))
    # Compute pi by drawing random (x, y) points and finding what
    # fraction lie inside a unit circle
    inside = 0
    for i in range(iterations):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if x ** 2 + y ** 2 <= 1.0:
            inside += 1

    out[thread_id] = 4.0 * inside / iterations


THR_PLANE = 0.01


@cuda.jit
def my_RANSAC(Ws, percents, segs, lens, start, rng_states):
    thread_id = cuda.grid(1)

    (i, j, k) = (cuda.threadIdx.x, cuda.blockIdx.x, cuda.blockDim.x)

    i1 = cuRand(rng_states, 0, lens[j])
    i2 = cuRand(rng_states, 0, lens[j])
    i3 = cuRand(rng_states, 0, lens[j])

    (p1, p2, p3) = (segs[start[j] + i1], segs[start[j] + i2], segs[start[j] + i3])
    w = Ws[j, i]  # cuda.local.array(shape=4, dtype=nb.float32)
    ux, uy, uz = cu_subtract_point(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
    vx, vy, vz = cu_subtract_point(p2[0], p2[1], p2[2], p3[0], p3[1], p3[2])
    w[0], w[1], w[2] = crossNormal(ux, uy, uz, vx, vy, vz)
    w[3] = -cu_dot(w[0], w[1], w[2], p1[0], p1[1], p1[2])

    cc = 0.0
    sum = 0
    for jj in range(lens[j]):
        p = segs[start[j] + jj]
        distance = math.fabs(cu_dot(w[0], w[1], w[2], p[0], p[1], p[2]) + w[3])
        sum = sum + distance
        if distance < THR_PLANE:
            cc = cc + 1
    percents[j, i] = cc / lens[j]

@jit(nopython=True)
def cross(a, b):
    c = np.ones_like(a)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    s = math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
    return c / s

@jit(nopython=True)
def cpu_RANSAC(Ws, percents, segs, lens, start):
    (N, M) = percents.shape
    for j in range(N):
        for i in range(M):
            (i1, i2, i3) = np.random.randint(0, lens[j], 3)
            (p1, p2, p3) = (segs[start[j] + i1], segs[start[j] + i2], segs[start[j] + i3])
            u = p1 - p2
            v = p2 - p3
            w = cross(u, v)
            d = -w.dot(p1)

            cc = 0
            for jj in range(lens[j]):
                p = segs[start[j] + jj]
                distance = math.fabs(w.dot(p) + d)
                if distance < THR_PLANE:
                    cc = cc + 1


            Ws[j, i, 0:3] = w
            Ws[j, i, 3] = d
            if d < 0:
                Ws[j,i] = -Ws[j,i]
            percents[j , i] = cc / lens[j]


def cpu_removeOutliers(Ws, percents, segs, lens, start):
    N = Ws.shape[0]
    segs = np.concatenate((segs, np.ones((segs.shape[0], 1))), axis=1)
    # P = np.split(segs, start)[1:]

    Pin = []
    for j in range(N):
        p = segs[start[j]:start[j] + lens[j]]
        distance = np.fabs(np.dot(p, Ws[j]))
        Pin.append(p[distance < THR_PLANE][:, 0:3])
    return Pin
