
import os
import h5py as h5
import numpy as np
import copy
from ..lib.io import MRC

from ..core import DataModel

import logging as log

def tomread(path, enhance=None, scale=None, dataset='data', norm=False):
    file_name, file_extension = os.path.splitext(path)
    data = None

    if file_extension in ['.rec', '.mrc']:
        data = MRC(path).data
    elif file_extension == '.npy':
        data = np.load(path)
    elif file_extension in ['.hdf5', '.h5']:
        with h5.File(path, 'r') as f:
            data = f[dataset][:]
    else:
        raise Exception('File format not supported')

    return data


def load_data(path=None, dataset='data'):
    log.info('+ Loading data to memory')
    data = tomread(path, dataset=dataset).astype(np.float32)

    log.info('+ Normalizing data')
    mean = data.mean()
    std = data.std()
    data -= mean
    data /= std

    log.info('+ Calculating data statistics')
    vmin, vmax = data.min(), data.max()
    evmin, evmax = np.percentile(data, (1., 99.))

    log.info('+ Writing data to disk')
    DM = DataModel.instance()
    params = dict(vmin=vmin, vmax=vmax, evmin=evmin, evmax=evmax,
                  default_evmin=evmin, default_evmax=evmax,
                  cmap='gray', active=True)
    DM.write_dataset('/data', data, params=params)
