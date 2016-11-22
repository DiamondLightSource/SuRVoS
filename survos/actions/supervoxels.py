
import h5py as h5
import numpy as np

import logging as log

from ..core import DataModel
from ..lib._superpixels import slic_supervoxels
from ..lib._rag import _unique_rag3d

DM = DataModel.instance()

def create_supervoxels(dataset=None, compactness=1., sp_shape=(5,10,10),
                       spacing=(5., 1., 1.), out_sv=None,
                       out_svindex=None, out_svtable=None,
                       out_svedges=None, out_svweights=None):

    log.info('+ Loading data into memory')
    data = DM.load_slices(dataset)

    log.info('+ Computing supervoxels')
    svlabels = slic_supervoxels(data, compactness=compactness,
                                sp_shape=sp_shape, spacing=spacing,
                                gpu=DM.selected_gpu)
    total_sv = svlabels.max()+1
    log.info('+ {} supervoxels created'.format(total_sv))

    log.info('+ Writing supervoxels to disk {}'.format(svlabels.shape))
    params = {
        'compactness': compactness,
        'sp_shape': sp_shape,
        'spacing': spacing,
        'num_supervoxels': total_sv,
        'source': dataset,
        'active': True
    }
    DM.create_empty_dataset(out_sv, shape=DM.data_shape, dtype=svlabels.dtype)
    DM.write_slices(out_sv, svlabels, params=params)

    log.info('+ Generating lookup table')
    svravel = svlabels.ravel()
    sortindexes = svravel.argsort()
    sortlabels = svravel[sortindexes]
    splits = np.searchsorted(sortlabels, np.arange(total_sv))
    if data.shape != DM.data_shape: # its a ROI, remap indexes
        log.info('+ Remaping indexes to ROI')
        zz, yy, xx = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
        zz += DM.active_roi[0].start
        yy += DM.active_roi[1].start
        xx += DM.active_roi[2].start
        seq = np.row_stack([zz.ravel(), yy.ravel(), xx.ravel()])
        idxtable = np.ravel_multi_index(seq, DM.data_shape)
        sortindexes = idxtable[sortindexes]

    log.info('+ Writing lookup table to disk {} {}'\
             .format(sortindexes.shape, splits.shape))

    DM.write_dataset(out_svindex, sortindexes, params=dict(active=True))
    DM.write_dataset(out_svtable, splits, params=dict(active=True))

    log.info('+ Generating supervoxel graph')
    edges = _unique_rag3d(svlabels, 6, total_sv)
    weights = np.ones(edges.shape[0], np.float32)

    log.info('+ Extracted {} unique edges'.format(edges.shape))
    DM.write_dataset(out_svedges, edges, params=dict(active=True))
    DM.write_dataset(out_svweights, weights, params=dict(active=True))

    return out_sv, total_sv, out_svindex, out_svtable, edges, weights
