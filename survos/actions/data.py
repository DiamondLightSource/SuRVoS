
import os
import h5py as h5
import numpy as np
import copy
import tifffile

from ..lib.io import MRC

from ..core import DataModel

import logging as log

def volread(path=None, dataset='data', stats=True, normalize=True):
    file_name, file_extension = os.path.splitext(path)
    data = None

    log.info('+ Loading data to memory')

    if file_extension in ['.rec', '.mrc']:
        data = np.copy(MRC(path).data)
    elif file_extension == '.npy':
        data = np.load(path)
    elif file_extension in ['.hdf5', '.h5']:
        with h5.File(path, 'r') as f:
            data = f[dataset][:]
    elif file_extension in ['.tif', '.tiff']:
        data = tifffile.imread(path, multifile=True)
    else:
        raise Exception('File format not supported')

    if normalize:
        log.info('+ Normalizing data')
        data = data.astype(np.float32)
        mean = data.mean()
        std = data.std()
        data -= mean
        data /= std

    if stats:
        log.info('+ Calculating data statistics')
        vmin, vmax = data.min(), data.max()
        evmin, evmax = np.percentile(data, (1., 99.))

        return data, vmin, vmax, evmin, evmax
    else:
        return data


def load_data(data=None, stats=None):
    vmin, vmax, evmin, evmax = stats
    log.info('+ Writing data to disk')
    DM = DataModel.instance()
    params = dict(vmin=vmin, vmax=vmax, evmin=evmin, evmax=evmax,
                  default_evmin=evmin, default_evmax=evmax,
                  cmap='gray', active=True)
    DM.write_dataset('/data', data, params=params)
