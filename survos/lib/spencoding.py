

import numpy as np

from ._spencoding import _sphist, _spmeans, _spstats


def normalize(features, norm='l1'):
    if norm in ['l1', 'hellinger']:
        features /= np.abs(features).sum(axis=1)[:, None]
    elif norm == 'l2':
        features /= np.sqrt((features**2).sum(axis=1))[:, None]
    elif norm == 'linf':
        features /= np.abs(features).max(axis=1)[:, None]
    elif norm == 'unit':
        features -= features.min(0)
        features /= features.max(0)
    if norm == 'hellinger':
        np.sqrt(features, features) # inplace
    return features


def sphist(data, splabels, nbins=100, norm=None):
    features = _sphist(data, splabels.flatten(), splabels.max()+1, nbins)
    return normalize(features, norm=norm)


def spmeans(data, splabels, norm=None):
    features =  _spmeans(data, splabels.flatten(), splabels.max()+1)
    return normalize(features, norm=norm)


def spstats(data, splabels, mode='append', sigmaset=False, covmode='full', norm=None):

    if mode not in ['append', 'add', None]:
        raise Exception('Only `append` or `add` methods are accepted')

    means, covars = _spstats(data, splabels.flatten(), splabels.max()+1)

    if sigmaset:
        # Add small constant to covars to make them positive-definite
        covars += np.eye(covars.shape[-1])[None, ...] * 1e-5
        covars = np.linalg.cholesky(covars) * np.sqrt(means.shape[1])

        if covmode == 'full':
            y1, x1 = np.tril_indices(means.shape[1], k=-1)
            y2, x2 = np.triu_indices(means.shape[1], k=1)
            covars[:, y2, x2] = covars[:, y1, x1]

    if mode == 'add':
        covars += means[:, :, None]

    if sigmaset and covmode == 'tril':
        y, x = np.tril_indices(means.shape[1])
        covars = covars[:, y, x]
    else:
        covars.shape = (covars.shape[0], -1)

    if mode == 'append':
        features = np.c_[means, covars]
    else:
        features = covars

    return normalize(features, norm=norm)
