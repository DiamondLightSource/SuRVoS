

import numpy as np
from skimage import color
from scipy import ndimage

from _features import _position_3d, _position_2d


class DensePositionExtractor(object):
    """Extracts position from images"""

    def __init__(self, unit=True, multichannel=False, **kwargs):
        self.unit = unit
        self.multichannel = multichannel

    def transform(self, X):
        if self.multichannel:
            if X.ndim == 3:
                G = _position_2d(X)
                G.shape = (-1, 2)
            else:
                G = _position_3d(X)
                G.shape = (-1, 3)
        else:
            if X.ndim == 2:
                G = _position_2d(X)
                G.shape = (-1, 2)
            else:
                G = _position_3d(X)
                G.shape = (-1, 3)

        if self.unit:
            G /= G.max(axis=0)

        return G


class DenseColorExtractor(object):
    """Extracts intensity features from images (grayscale or color)"""

    def __init__(self, multichannel=False, convert2lab=False, **kwargs):
        self.multichannel = multichannel
        self.convert2lab = convert2lab

    def transform(self, X):
        if self.multichannel and self.convert2lab:
            X = color.rgb2lab(X)

        if self.multichannel:
            return X.reshape(-1, X.shape[-1])
        else:
            return X.reshape(-1, 1)


class DenseTextureExtractor(object):
    """Extracts local texture features from images"""

    def __init__(self, multichannel=False, order=2, sigmas=[1., 2.], **kwargs):
        self.multichannel = multichannel
        self.order = order
        self.sigmas = sigmas

    def transform(self, X):
        f = []
        for sigma in self.sigmas:
            g = ndimage.gaussian_filter(X, sigma)
            dg = np.gradient(g)
            g.shape = g.shape + (1,)
            for d in dg:
                d.shape = d.shape + (1,)
            f += [g,] + dg
        f = np.concatenate(f, axis=-1)
        f.shape = (-1, f.shape[-1])
        return f


class DensePatchExtractor2D(object):

    def __init__(self, patch_size=(21,21), flatten=True):
        self.patch_size = patch_size
        self.flatten = flatten

    def transform(self, img):
        ypad, xpad = self.patch_size[0]//2, self.patch_size[1]//2
        padded = np.pad(img, ((ypad, ypad), (xpad, xpad)), mode='reflect')
        Y, X = padded.shape
        y, x = self.patch_size
        shape = ((Y-y+1), (X-x+1), y, x) # number of patches, patch_shape
        # The right strides can be thought by:
        # 1) Thinking of `img` as a chunk of memory in C order
        # 2) Asking how many items through that chunk of memory are needed when indices
        #    i,j,k,l are incremented by one
        strides = padded.itemsize*np.array([X, 1, X, 1])
        if self.flatten:
            return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).reshape(-1, self.patch_size[0] * self.patch_size[1])
        else:
            return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)


class DensePatchExtractor3D(object):

    def __init__(self, patch_size=21, flatten=False, pad=True):
        self.patch_size = patch_size
        self.flatten = flatten
        self.pad = pad

    def transform(self, img):
        if self.pad:
            padded = np.pad(img, self.patch_size//2, mode='reflect')
        else:
            padded = img.copy()
        Z, Y, X = padded.shape
        z, y, x = self.patch_size, self.patch_size, self.patch_size
        shape = ((Z-z+1), (Y-y+1), (X-x+1), z, y, x) # number of patches, patch_shape
        strides = padded.itemsize*np.array([Y*X, X, 1, Y*X, X, 1])
        if self.flatten:
            return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).reshape(-1, self.patch_size[0] * self.patch_size[1])
        else:
            return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
