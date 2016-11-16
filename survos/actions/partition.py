

import numpy as np

import logging as log

from ..core import DataModel

from scipy.ndimage.measurements import label as splabel
from skimage.morphology import octahedron, ball

from scipy.ndimage import measurements as measure
from scipy.stats import binned_statistic
from sklearn.decomposition import PCA

from ..lib._zernike import zernike_descriptors


def label_objects(dataset=None, labels=None, out=None, out_features=None,
				  source=None, return_labels=False):
	DM = DataModel.instance()

	log.info('+ Loading data into memory')
	data = DM.load_slices(dataset)
	if labels is None:
		data += 1
		labels = set(np.unique(data)) - set([0])
	else:
		data += 1
		labels = np.asarray(labels) + 1

	obj_labels = []

	log.info('+ Extracting individual objects')
	new_labels = np.zeros(data.shape, np.int32)
	total_labels = 0
	num = 0

	for label in labels:
		mask = (data == label)
		tmp_data = data.copy()
		tmp_data[~mask] = 0
		tmp_labels, num = splabel(tmp_data, structure=octahedron(1))
		mask = (tmp_labels > 0)
		new_labels[mask] = tmp_labels[mask] + total_labels
		total_labels += num
		obj_labels += [label] * num

	log.info('+ {} Objects found'.format(total_labels))
	log.info('+ Saving results')
	DM.create_empty_dataset(out, DM.data_shape, new_labels.dtype)
	DM.write_slices(out, new_labels, params=dict(active=True, num_objects=total_labels))

	log.info('+ Loading source to memory')
	data = DM.load_slices(source)
	objs = new_labels
	objects = new_labels
	num_objects = total_labels
	objlabels = np.arange(1, num_objects+1)

	log.info('+ Computing Average intensity')
	feature = measure.mean(data, objs, index=objlabels)
	DM.create_empty_dataset(out_features[0], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[0], feature, params=dict(active=True))

	"""log.info('+ Computing Median intensity')
	objs.shape = -1
	data.shape = -1
	feature = binned_statistic(objs, data, statistic='median',
							   bins=num_objects+1)[0]
	feature = feature[objlabels]
	out_features[1].write_direct(feature)
	out_features[1].attrs['active'] = True
	objs.shape = dataset.shape
	data.shape = dataset.shape"""

	log.info('+ Computing Sum of intensity')
	feature = measure.sum(data, objs, index=objlabels)
	DM.create_empty_dataset(out_features[1], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[1], feature, params=dict(active=True))

	log.info('+ Computing Standard Deviation of intensity')
	feature = measure.standard_deviation(data, objs, index=objlabels)
	DM.create_empty_dataset(out_features[2], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[2], feature, params=dict(active=True))

	log.info('+ Computing Variance of intensity')
	feature = measure.variance(data, objs, index=objlabels)
	DM.create_empty_dataset(out_features[3], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[3], feature, params=dict(active=True))

	log.info('+ Computing Area')
	objs.shape = -1
	feature = np.bincount(objs, minlength=num_objects+1)[1:]
	DM.create_empty_dataset(out_features[4], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[4], feature, params=dict(active=True))
	DM.create_empty_dataset(out_features[5], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[5], np.log10(feature), params=dict(active=True))
	objs.shape = data.shape

	log.info('+ Computing Bounding Box')
	obj_windows = measure.find_objects(objs)
	feature = []; depth = []; height = []; width = [];
	for w in obj_windows:
		feature.append((w[0].stop - w[0].start) *
					   (w[1].stop - w[1].start) *
					   (w[2].stop - w[2].start))
		depth.append(w[0].stop - w[0].start)
		height.append(w[1].stop - w[1].start)
		width.append(w[2].stop - w[2].start)

	feature = np.asarray(feature, np.float32)
	DM.create_empty_dataset(out_features[6], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[6], feature, params=dict(active=True))
	#depth
	depth = np.asarray(depth, np.float32)
	DM.create_empty_dataset(out_features[7], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[7], depth, params=dict(active=True))
	# height
	height = np.asarray(height, np.float32)
	DM.create_empty_dataset(out_features[8], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[8], height, params=dict(active=True))
	# width
	width = np.asarray(width, np.float32)
	DM.create_empty_dataset(out_features[9], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[9], width, params=dict(active=True))
	# log10
	DM.create_empty_dataset(out_features[10], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[10], np.log10(feature), params=dict(active=True))

	log.info('+ Computing Oriented Bounding Box')
	ori_feature = []; ori_depth = []; ori_height = []; ori_width = [];
	for i, w in enumerate(obj_windows):
		z, y, x = np.where(objs[w] == i+1)
		coords = np.c_[z, y, x]
		if coords.shape[0] >= 3:
			coords = PCA(n_components=3).fit_transform(coords)
		cmin, cmax = coords.min(0), coords.max(0)
		zz, yy, xx = (cmax[0] - cmin[0] + 1,
					  cmax[1] - cmin[1] + 1,
					  cmax[2] - cmin[2] + 1)
		ori_feature.append(zz * yy * xx)
		ori_depth.append(zz)
		ori_height.append(yy)
		ori_width.append(xx)

	ori_feature = np.asarray(ori_feature, np.float32)
	DM.create_empty_dataset(out_features[11], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[11], ori_feature, params=dict(active=True))
	#depth
	ori_depth = np.asarray(ori_depth, np.float32)
	DM.create_empty_dataset(out_features[12], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[12], ori_depth, params=dict(active=True))
	# height
	ori_height = np.asarray(ori_height, np.float32)
	DM.create_empty_dataset(out_features[13], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[13], ori_height, params=dict(active=True))
	# width
	ori_width = np.asarray(ori_width, np.float32)
	DM.create_empty_dataset(out_features[14], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[14], ori_width, params=dict(active=True))
	# log10
	DM.create_empty_dataset(out_features[15], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[15], np.log10(ori_feature), params=dict(active=True))

	log.info('+ Computing Positions')
	pos = measure.center_of_mass(objs, labels=objs, index=objlabels)
	pos = np.asarray(pos, dtype=np.float32)
	DM.create_empty_dataset(out_features[16], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[16], pos[:, 2].copy(), params=dict(active=True))
	DM.create_empty_dataset(out_features[17], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[17], pos[:, 1].copy(), params=dict(active=True))
	DM.create_empty_dataset(out_features[18], (num_objects,), np.float32, check=False)
	DM.write_dataset(out_features[18], pos[:, 0].copy(), params=dict(active=True))

	if return_labels:
		return out, total_labels, np.asarray(obj_labels)

	return out, total_labels


def apply_rules(features=None, label=None, rules=None, out_ds=None, num_objects=None):
	DM = DataModel.instance()

	mask = np.ones(num_objects, dtype=np.bool)
	out = DM.load_ds(out_ds)
	out[out == label] = -1

	for f, s, t in rules:
		if s == 0:
			np.logical_and(mask, (DM.load_ds(features[f]) > t), out=mask)
		else:
			np.logical_and(mask, (DM.load_ds(features[f]) < t), out=mask)

	out[mask] = label
	DM.write_dataset(out_ds, out)
	return out
