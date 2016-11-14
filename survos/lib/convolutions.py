

import numpy as np

# 2D Dosition
from scipy.linalg import svd
# 3D Dosition
from sktensor import dtensor, cp_als

from ._convolutions import gconv, gconvs, gconvssh, ngconvssh


def make_rotated_grid(shape, orient):
    r = [s//2 for s in shape]
    r = [(-r[i], r[i]+1) if 2 * r[i] + 1 == shape[i] else (-r[i], r[i]) \
         for i in range(len(r))]
    z, y, x = np.mgrid[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1]]

    orgpts = np.vstack([z.ravel(), y.ravel(), x.ravel()])
    a, b = orient
    rotmz = [
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a), np.cos(a)]
    ]
    rotmy = [
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ]
    rotpts = np.dot(rotmy, np.dot(rotmz, orgpts))
    return [rotpts[i, :].reshape(z.shape) for i in range(3)]


def make_gaussian_1d(sigma=1., size=None, order=0, trunc=3):
    if size == None:
        size = sigma * trunc * 2 + 1
    x = np.arange(-(size//2), (size//2)+1)
    if order > 2:
        raise ValueError("Only orders up to 2 are supported")
    # compute unnormalized Gaussian response
    response = np.exp(-x ** 2 / (2. * sigma ** 2))
    if order == 1:
        response = -response * x
    elif order == 2:
        response = response * (x ** 2 - sigma ** 2)
    # normalize
    response /= np.abs(response).sum()
    return response.astype(np.float32)


def make_3d_gaussian(shape, sigma, orders=(0, 1, 0), orient=(0,0)):
    if type(sigma) == tuple or type(sigma) == list:
        sz, sy, sx = sigma
    else:
        sz, sy, sx = (sigma * (1. + 2. * o) for o in orders)
    rotz, roty, rotx = make_rotated_grid(shape, orient)

    g = np.zeros(shape, dtype=np.float32)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sx ** 2 + roty ** 2 / sy ** 2 + rotz ** 2 / sz ** 2))
    g /= 2 * np.pi * sx * sy * sz

    for o, s, x in zip(orders, (sz, sy, sx), (rotz, roty, rotx)):
        if o == 1:
            g *= -x
        elif o == 2:
            g *= (x**2 - s**2)

    return g

def make_3d_gabor(shape, sigmas, frequency, offset=0, orient=(0,0), return_real=True):
    sz, sy, sx = sigmas
    rotz, roty, rotx = make_rotated_grid(shape, orient)

    g = np.zeros(shape, dtype=np.complex)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sx ** 2 + roty ** 2 / sy ** 2 + rotz ** 2 / sz ** 2))
    g /= 2 * np.pi * sx * sy * sz
    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    if return_real:
        return np.real(g)

    return g


def rec_2d(D, weights=None):
    if weights is None:
        weights = np.ones(len(D), D[0][0].dtype)
    R = np.zeros((D[0][0].shape[0], D[0][1].shape[0]))
    for i in range(len(D)):
        R += D[i][0][:, None] * D[i][1][None, :] * weights[i]
    return R

def rec_3d(D, weights=None):
    if weights is None:
        weights = np.ones(len(D), D[0][0].dtype)
    R = np.zeros((D[0][0].shape[0], D[0][1].shape[0], D[0][2].shape[0]))
    for i in range(len(D)):
        R += D[i][0][:, None, None] * D[i][1][None, :, None] * D[i][2][None, None, :] * weights[i]
    return R

def rec_error(X, R, mean=True):
    if mean:
        return np.mean((X - R)**2)
    else:
        return np.sum((X - R)**2)

def separate_kernel(kernel, max_rank=1, return_error=False, return_weights=False):
    if kernel.ndim == 1:
        return kernel
    elif kernel.ndim == 2:
        U, s, V = svd(kernel)

        mrank = np.linalg.matrix_rank(kernel)
        rank = max_rank if max_rank is not None and max_rank <= mrank  else mrank
        weights = s[:rank]

        if return_weights:
            D = [(U[:, i], V[i, :]) for i in range(rank)]
            if return_error:
                return D, weights, rec_error(kernel, rec_2d(D, weights))
            else:
                return D, weights
        else:
            D = [(U[:, i] * weights[i], V[i, :]) for i in range(rank)]
            if return_error:
                return D, rec_error(kernel, rec_2d(D))
            else:
                return D

    elif kernel.ndim == 3:
        rank = min(max_rank, min(*kernel.shape))
        T = dtensor(kernel)
        P, fit, itr, exectimes = cp_als(T, rank, init='random')

        Uz, Uy, Ux = P.U
        weights = P.lmbda

        if return_weights:
            D = [(Uz[:, i], Uy[:, i], Ux[:, i]) for i in range(rank)]
            if return_error:
                return D, weights, rec_error(kernel, rec_3d(D, weights))
            else:
                return D, weights
        else:
            D = [(Uz[:, i] * weights[i], Uy[:, i], Ux[:, i]) for i in range(rank)]
            if return_error:
                return D, rec_error(kernel, rec_3d(D))
            else:
                return D
    else:
        raise Exception("Kernel dimensions not supported")
