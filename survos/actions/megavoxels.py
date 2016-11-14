
import h5py as h5
import numpy as np

import logging as log

from ..core import DataModel
from ..lib.rag import create_rag
from ..lib.spencoding import spmeans
from ..lib.supersegments import HPC

DM = DataModel.instance()

def create_megavoxels(dataset=None, splabels=None, num_sps=None,
					  lamda=0.1, nbins=10, gamma=None, source=1,
					  out_mv=None, out_mvindex=None, out_mvtable=None):

	log.info('# Params: (num_sps: {}, lamda: {}, nbins: {}, gamma: {})'
			 .format(num_sps, lamda, nbins, gamma))

	log.info('+ Loading data into memory')
	data = DM.load_slices(dataset)
	dshape = data.shape
	data.shape = (-1, 1)

	log.info('+ Loading supervoxels into memory')
	supervoxels = DM.load_slices(splabels)
	spsize = num_sps

	min_boundary = int(round(np.bincount(supervoxels.ravel()).mean() ** (1./3.)))
	log.info('+ Creating supervoxel graph: MinBound = {0:.2f}'.format(min_boundary))
	edges = create_rag(supervoxels, connectivity=6, min_boundary=10,
					   return_rag=False, return_counts=False)

	log.info('+ Creating supervoxel intensity features')
	data = spmeans(data, supervoxels)

	if gamma is not None:
		gamma = 1 / (2. * (gamma**2))

	log.info('+ Creating megavoxels')
	mvlabels = HPC(data, edges, lamda=lamda, nbow=nbins,
				   label_potential='l1', gamma=gamma,
				   min_size=None, max_size=spsize, verbose=False,
				   max_iter=2, nsamples=min(10000,spsize))

	total_mv = mvlabels.max()+1
	log.info('+ {} megavoxels created'.format(total_mv))

	mvlabels = mvlabels[supervoxels]

	log.info('+ Writing megavoxels to disk {}'.format(mvlabels.shape))
	params = {
		'lamda': lamda,
		'nbins': nbins,
		'gamma': 'None' if gamma is None else gamma,
		'num_megavoxels': total_mv,
		'source': dataset,
		'active': True
	}
	DM.create_empty_dataset(out_mv, shape=DM.data_shape, dtype=mvlabels.dtype)
	DM.write_slices(out_mv, mvlabels, params=params)

	log.info('+ Generating lookup table')
	mvravel = mvlabels.ravel()
	sortindexes = mvravel.argsort()
	sortlabels = mvravel[sortindexes]
	splits = np.searchsorted(sortlabels, np.arange(total_mv))

	if data.shape != DM.data_shape: # its a ROI, remap indexes
		log.info('+ Remaping indexes to ROI')
		zz, yy, xx = np.mgrid[:dshape[0], :dshape[1], :dshape[2]]
		zz += DM.active_roi[0].start
		yy += DM.active_roi[1].start
		xx += DM.active_roi[2].start
		seq = np.row_stack([zz.ravel(), yy.ravel(), xx.ravel()])
		idxtable = np.ravel_multi_index(seq, DM.data_shape)
		sortindexes = idxtable[sortindexes]

	log.info('+ Writing lookup table to disk {} {}'\
			 .format(sortindexes.shape, splits.shape))
	DM.write_dataset(out_mvindex, sortindexes, params=dict(active=True))
	DM.write_dataset(out_mvtable, splits, params=dict(active=True))

	return out_mv, total_mv, out_mvindex, out_mvtable
