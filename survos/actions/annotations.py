

import numpy as np
import networkx as nx
import logging as log

from ..lib.convolutions import make_3d_gaussian

from scipy.ndimage.measurements import label as splabel
from scipy.ndimage.morphology import binary_closing, binary_dilation, \
                                     binary_erosion, binary_opening, \
                                     binary_fill_holes
from skimage.morphology import diamond, octahedron
from skimage.draw import circle

from ..core import DataModel


DM = DataModel.instance()

def refine_label(data=None, label=None, method=None, radius=1, slide=None):
    ds = data
    data = DM.load_slices(ds)
    rshape = DM.region_shape()

    zmin = DM.active_roi[0].start
    ymin = DM.active_roi[1].start
    xmin = DM.active_roi[2].start

    if method == 'fill_holes':
        radius = 1

    if type(slide) == str:
        mask = (data == label)
        if slide == '3D':
            msize = max(rshape)
            selem = octahedron(radius)
        else:
            msize = max(rshape[1:])
            selem = diamond(radius)
    else:
        msize = max(rshape[1:])
        mask = (data[slide] == label)
        selem = diamond(radius)

    if radius > np.sqrt(msize):
        raise Exception('Radius too large')

    funcs = {
        'dilation' : binary_dilation,
        'erosion' : binary_erosion,
        'opening' : binary_opening,
        'closing' : binary_closing,
        'fill_holes' : binary_fill_holes
    }

    if method not in funcs:
        log.info('+ Refinement {} not supported'.format(method))
        return None

    log.info('+ Performing {} refinement ({})'.format(method, slide))
    if slide != '2D' and mask.any():
        result = funcs[method](mask, structure=selem)
    else:
        result = np.zeros(data.shape, dtype=np.bool)
        f = funcs[method]
        for i in range(data.shape[0]):
            if mask[i].any():
                result[i] = f(mask[i], structure=selem)

    log.info('+ Calculating changes..')
    changes = (result != mask)
    if type(slide) == str:
        values = data[changes]
        data[mask] = -1
        data[result] = label
        changes = np.column_stack(np.where(changes)) + np.array([zmin, ymin, xmin], np.int32)
    else:
        values = data[slide, changes]
        data[slide, mask] = -1
        data[slide, result] = label
        changes = np.column_stack(np.where(changes[None, ...])) + np.array([zmin+slide, ymin, xmin], np.int32)

    DM.write_slices(ds, data)

    log.info('+ done.')
    return changes, values



def save_threshold(source=None, vmin=None, vmax=None, level=None, label=None):
    data = DM.load_slices(source)
    target = DM.load_slices(level)
    mask = (data > vmin) & (data < vmax)
    target[mask] = label
    DM.write_slices(level, target)