

import numpy as np
from ..qt_compat import QtGui, QtCore

from Queue import Queue

from .base import Plugin
from ..widgets import HWidgets, LCheckBox, RCheckBox, PLineEdit, \
					  HeaderLabel, SubHeaderLabel, TComboBox, \
					  RoundedWidget, HSize3D, SourceCombo, BLabel, \
					  ActionButton, SectionCombo
from ..core import DataModel, LayerManager, Launcher
from .. import actions as ac

import logging as log

class Int3d: pass
class Float3d: pass

class Choice(object):
	def __init__(self, options):
		self.options = options

available_features = [
	('Raw', None, None),
	('Threshold', 'thresh', []),
	('Invert Threshold', 'inv_thresh', []),
	##################################################
	('Denoising', None, None),
	('Gaussian Filter', 'gauss', [
		('Sigma', Float3d, 2.),
	]),
	('Total Variation', 'tv', [
		('Spacing', Float3d, 1.),
		('Lambda', float, 10.),
		('Max Iter', int, 100)
	]),
	##################################################
	('Local Features', None, None),
	('Local Mean', 'local_mean', [
		('Radius', Int3d, 4),
	]),
	('Local Standard Deviation', 'local_std', [
		('Radius', Int3d, 4),
	]),
	('Local Centering', 'local_center', [
		('Radius', Int3d, 4),
	]),
	('Local Normalization', 'local_norm', [
		('Radius', Int3d, 4),
	]),
	('Local Gradient Magnitude', 'local_mag', [
		('Radius', Int3d, 4),
	]),
	('Local Gradient Orientation', 'local_ori', [
		('Radius', Int3d, 4),
	]),
	##################################################
	('Gaussian Features', None, None),
	('Gaussian Centering', 'gauss_center', [
		('Sigma', Float3d, 2.),
	]),
	('Gaussian Normalization', 'gauss_norm', [
		('Sigma', Float3d, 2.),
	]),
	('Gaussian Gradient Magnitude', 'gauss_mag', [
		('Sigma', Float3d, 2.),
	]),
	('Gaussian Gradient Orientation', 'gauss_ori', [
		('Radius', Int3d, 3),
		('Sigma', Float3d, 2.),
	]),
	##################################################
	('Blob Detection', None, None),
	('Difference of Gaussian', 'dog', [
		('Sigma', Float3d, 2.),
		('Sigma Ratio', float, 1.6),
		('Threshold', bool, False),
		('Response', Choice(['Bright', 'Dark']), 'Bright'),
	]),
	('Laplacian of Gaussian', 'log', [
		('Sigma', Float3d, 2.),
		('Threshold', bool, False),
		('Response', Choice(['Bright', 'Dark']), 'Bright')
	]),
	('Determinant of Hessian', 'hessian_det', [
		('Sigma', Float3d, 2.),
	]),
	('Determinant of Struct. Tensor', 'structure_det', [
		('Sigma Deriv', Float3d, 1.),
		('Sigma Area', Float3d, 3.),
	]),
	##################################################cd
	('Texture and Structure', None, None),
	('Hessian Eigenvalues', 'hessian_eig', [
		('Sigma', Float3d, 1.),
		('Eigen Value', Choice([0,1,2]), 0),
	]),
	('Struct. Tensor Eigenvalues', 'structure_eig', [
		('Sigma Deriv', Float3d, 1.),
		('Sigma Area', Float3d, 3.),
		('Eigen Value', Choice([0,1,2]), 0),
	]),
	('Gabor Filter', 'gabor', [
		('Sigma Deriv', Float3d, 2.),
		('Sigma Area', Float3d, 3.),
		('Eigen Value', Choice([0,1,2]), 0),
	]),
	##################################################
	('Robust Features', None, None),
	('(SI) Gaussian', 'gauss_scale3d', [
		('Init Sigma', float, 1),
		('Sigma Incr', float, 2.),
		('Max Sigma', float, 10.),
		('Response', Choice(['Min', 'Max', 'Avg']), 'Max')
	]),
	('Derivative Rotation Invariant', 'gauss_ori3d', [
		('Sigma', Float3d, 3.),
		('Num Orientations', Choice([2,4,8,16]), 8),
		('Response', Choice(['Min', 'Max', 'Avg']), 'Max')
	]),
	('(SI) Difference of Gaussians', 'si_dog', [
		('Init Sigma', Float3d, 1),
		('Sigma Ratio', float, 1.6),
		('Max Sigma', float, 15.),
		('Threshold', bool, False),
		('Response', Choice(['Bright', 'Dark']), 'Bright')
	]),
	('(SI) Laplacian of Gaussian', 'si_laplacian', [
		('Init Sigma', Float3d, 1),
		('Sigma Incr', float, 2.),
		('Max Sigma', float, 10.),
		('Threshold', bool, False),
		('Response', Choice(['Bright', 'Dark']), 'Bright')
	]),
	('(SI) Determinant of Hessian', 'si_hessian_det', [
		('Init Sigma', Float3d, 1),
		('Sigma Incr', float, 2.),
		('Max Sigma', float, 10.),
		('Response', Choice(['Bright', 'Dark']), 'Bright')
	]),
	('(SI) Frangi Filter', 'frangi', [
		('Init Sigma', Float3d, 1),
		('Sigma Incr', float, 2.),
		('Max Sigma', float, 10.),
		('Lamda', float, 0.5),
		('Response', Choice(['Bright', 'Dark']), 'Bright')
	]),
	##################################################
	('Activation Layers', None, None),
	('Maximum Response', 'max_response', []),
	('Rectified Linear Unit', 'relu', [
		('Type', Choice(['Standard', 'Noisy', 'Leaky']), 'Standard'),
		('alpha', float, 0.01)
	]),
]

available_descriptors = [
	'Mean',
	'Feature Histogram',
	'Texton Histogram',
	'Covariance',
	'SigmaSet'
]

class FeatureRow(RoundedWidget):

	toggled = QtCore.pyqtSignal(int, bool)
	delete = QtCore.pyqtSignal(int)
	compute = QtCore.pyqtSignal(int, str, str, str, dict)

	def __init__(self, idx, ftype=1, fname=None, active=None,
				 load_params={}, parent=None):
		super(FeatureRow, self).__init__(parent=parent, color=None,
										 bg='#cde5e5', width=0)

		self.DM = DataModel.instance()

		self.idx = idx
		self.feature = available_features[ftype]
		if fname is None:
			self.fname = '{}_{}'.format(idx, self.feature[1])
		else:
			self.fname = fname
		self.dsname = 'channels/{}'.format(self.fname)
		params = dict(active=False, feature_idx=self.idx,
					  feature_type=self.feature[1], feature_name=self.fname)
		if len(params) > 0:
			self.DM.create_empty_dataset(self.dsname, shape=self.DM.data_shape,
										 dtype=np.float32, params=params,
										 fillvalue=np.nan)

		self.chk_compute = QtGui.QCheckBox()
		self.btn_delete = QtGui.QPushButton('X')
		self.btn_delete.setMaximumWidth(25)
		self.btn_delete.setMaximumHeight(30)
		self.btn_delete.setMinimumHeight(30)
		self.btn_delete.setMinimumWidth(25)

		self.txt_name = QtGui.QLabel('[{}] {}'.format(self.idx, self.feature[0]))
		self.txt_name.setStyleSheet('background-color: #6DC7C7; color: #006161;'
									'padding-left: 10px;')

		self.btn_compute = QtGui.QPushButton(u'\u2713')
		self.btn_compute.setContentsMargins(0,0,0,0)
		self.btn_compute.setMaximumWidth(25)
		self.btn_compute.setCheckable(True)
		if active == True:
			self.btn_compute.setChecked(True)

		self.btn_show = QtGui.QPushButton(u'\u25B2')
		self.btn_show.setMaximumWidth(25)
		self.btn_show.setMaximumHeight(30)
		self.btn_show.setMinimumHeight(30)
		self.btn_show.setMinimumWidth(25)
		self.visible = True

		vbox = QtGui.QVBoxLayout()
		self.setLayout(vbox)
		vbox.addWidget(HWidgets(self.chk_compute, self.btn_delete,
								self.txt_name,
								self.btn_compute, self.btn_show,
								stretch=[0,0,1,0,0]))

		self.txt_name.mousePressEvent = self.toggleParams
		self.btn_show.clicked.connect(self.toggleParams)
		self.btn_delete.clicked.connect(self.on_delete)
		self.btn_compute.clicked.connect(self.on_compute)

		self.param_container = QtGui.QWidget()
		vbox.addWidget(self.param_container)

		vbox = QtGui.QVBoxLayout(self.param_container)
		vbox.setContentsMargins(0,0,0,0)

		self.source_combo = SourceCombo(ignore=self.fname) # ignore self
		self.clamp_chk = LCheckBox('Clamp')
		if 'clamp' in load_params:
			self.clamp_chk.setChecked(load_params['clamp'])
		else:
			self.clamp_chk.setChecked(False)
		if available_features[ftype][1] == 'thresh':
			self.clamp_chk.setChecked(True)
			self.clamp_chk.setEnabled(False)
		if 'source' in load_params:
			self.source_combo.loadSource(load_params['source'])
		vbox.addWidget(HWidgets(self.clamp_chk, None, self.source_combo,
								stretch=[0,1,0,0]))

		self.isChecked = self.chk_compute.isChecked
		self.setChecked = self.chk_compute.setChecked

		self.params = []
		self.init_params(vbox, load_params)
		self.showParams(False)

	def init_params(self, vbox, load_params):
		for param in self.feature[2]:
			if param[1] == bool:
				curr = QtGui.QCheckBox()
				curr.setChecked(param[2])
				curr.value = curr.isChecked
				if param[0] in load_params:
					curr.setChecked(load_params[param[0]])
			elif param[1] == Float3d:
				curr = HSize3D('', default=(param[2],param[2],param[2]),
							   parse=float, txtwidth=10, coordwidth=50)
				if param[0] in load_params:
					curr.setValue(load_params[param[0]])
			elif param[1] == Int3d:
				curr = HSize3D('', default=(param[2],param[2],param[2]),
							   parse=int, txtwidth=10, coordwidth=50)
				if param[0] in load_params:
					curr.setValue(load_params[param[0]])
			elif type(param[1]) == Choice:
				curr = TComboBox('', param[1].options, parse=type(param[1].options[0]))
				curr.setCurrentIndex(param[1].options.index(param[2]))
				if param[0] in load_params:
					curr.setCurrentIndex(param[1].options.index(load_params[param[0]]))
			else:
				curr = PLineEdit(parse=param[1], default=param[2])
				if param[0] in load_params:
					curr.setText(str(load_params[param[0]]))

			vbox.addWidget(HWidgets(None, param[0], curr, stretch=[1,0,0]))
			self.params.append((param[0], curr))

	def showParams(self, visible):
		self.visible = visible
		if self.visible:
			self.btn_show.setText(u'\u25BC')
		else:
			self.btn_show.setText(u'\u25B2')
		self.param_container.setVisible(self.visible)

	def toggleParams(self, dummy):
		self.showParams(not self.visible)
		self.toggled.emit(self.idx, self.visible)

	def on_delete(self):
		self.delete.emit(self.idx)

	def get_params(self):
		params = {k: v.value() for k, v in self.params}
		params['source'] = self.source_combo.value()
		params['clamp'] = self.clamp_chk.isChecked()
		return params

	def on_compute(self):
		params = self.get_params()
		self.compute.emit(self.idx, self.txt_name.text(), self.fname,
						  self.feature[1], params)

	def setComputed(self, bol):
		self.btn_compute.setChecked(bol)


class FeatureChannels(Plugin):

	name = 'Feature Channels'

	def __init__(self, parent=None):
		super(FeatureChannels, self).__init__(ptype=Plugin.Plugin, parent=parent)

		self.DM = DataModel.instance()
		self.LM = LayerManager.instance()
		self.launcher = Launcher.instance()

		self.addWidget(HeaderLabel('Feature Channels'))
		#self.cmb_features = QtGui.QComboBox(self)
		self.cmb_features = SectionCombo('Select Feature', parent=self)
		self.cmb_features.setMinimumWidth(280)
		for i, feature in enumerate(available_features):
			self.cmb_features.addItem(feature[0], section=feature[2] is None)

		self.btn_add = ActionButton('Add')

		dummy = QtGui.QWidget(self)
		vbox = QtGui.QVBoxLayout(dummy)
		dummy.setLayout(vbox)
		vbox.addWidget(HWidgets(self.cmb_features, self.btn_add, stretch=[1,0],
								parent=self))

		self.chk_all = QtGui.QCheckBox("Select all features")
		self.chk_all.stateChanged.connect(self.on_check_all)
		self.cmp_all = QtGui.QPushButton("Compute features")
		self.cmp_all.setMinimumWidth(150)
		self.cmp_all.clicked.connect(self.on_compute_all)
		vbox.addWidget(HWidgets(self.chk_all, None, self.cmp_all, stretch=[0,1,0]))

		self.addWidget(dummy)

		spacer = QtGui.QWidget()
		spacer.setMinimumHeight(10)
		spacer.setMaximumHeight(10)
		vbox.addWidget(spacer)

		vbox.addWidget(SubHeaderLabel('Available Channels'))

		dummy = QtGui.QWidget()
		self.feature_container = QtGui.QVBoxLayout()
		dummy.setLayout(self.feature_container)
		self.addWidget(dummy)

		self.btn_add.clicked.connect(self.on_add_feature)

		self.features = {}
		self.num_features = 0

	def ftype2idx(self, ftype):
		for i, f in enumerate(available_features):
			if f[1] == ftype:
				return i
		return -1

	def load_channel(self, fidx, fname, ftype, active, params):
		ftype = self.ftype2idx(ftype)
		if ftype < 0:
			log.info('[[[Error]]] feature {} could not be loaded'.format(fname))
			return
		feature_row = FeatureRow(fidx, ftype, fname=fname, active=active,
								 load_params=params, parent=self)
		feature_row.toggled.connect(self.on_toggle_feature)
		feature_row.delete.connect(self.on_delete_feature)
		feature_row.compute.connect(self.on_compute_feature)
		self.feature_container.addWidget(feature_row)

		self.features[fidx] = feature_row
		self.on_toggle_feature(-1, True)

		if self.num_features <= fidx:
			self.num_features = fidx + 1

	def on_add_feature(self):
		ftype = self.cmb_features.currentIndex()
		feature_row = FeatureRow(self.num_features, ftype, parent=self)
		feature_row.toggled.connect(self.on_toggle_feature)
		feature_row.delete.connect(self.on_delete_feature)
		feature_row.compute.connect(self.on_compute_feature)

		self.feature_container.addWidget(feature_row)

		self.features[self.num_features] = feature_row
		self.on_toggle_feature(-1, True)

		self.num_features += 1

	def on_toggle_feature(self, idx, visible):
		if visible:
			for k, v in self.features.items():
				if k != idx:
					v.showParams(False)

	def on_delete_feature(self, idx):
		channel_name = self.features[idx].dsname
		self.DM.remove_dataset(channel_name)
		self.features[idx].setParent(None)
		self.DM.channel_removed.emit(channel_name)
		if idx == self.num_features - 1:
			self.num_features -= 1

	def on_check_all(self, state):
		check = (state == QtCore.Qt.Checked)
		for feature in self.features.values():
			feature.setChecked(check)

	def on_compute_feature(self, idx, feature_name, feature_id, feature_desc, params):
		source = params['source']
		clamp = None
		if params['clamp'] == True:
			attrs = self.DM.attrs(source)
			clamp = attrs['evmin'], attrs['evmax']
		out = 'channels/{}'.format(feature_id)

		self.current_name = out

		self.launcher.run(ac.compute_channel, name=feature_name,
						  source=source, clamp=clamp, out=out, idx=idx,
						  feature=feature_desc, params=params,
						  caption='Computing Feature: {}'.format(feature_desc),
						  cb=self.on_features_computed)

	def on_compute_all(self):
		features = []
		for k, feature in self.features.items():
			if not feature.isChecked():
				continue
			params = feature.get_params()
			source = params['source']
			clamp = None
			if params['clamp'] == True:
				attrs = self.DM.attrs(source)
				clamp = attrs['evmin'], attrs['evmax']
			out = feature.dsname
			idx = feature.idx
			feature_name = feature.txt_name.text()
			feature_desc = feature.feature[1]

			features.append({
				'name'          : feature_name,
				'source'        : source,
				'clamp'         : clamp,
				'params'        : params,
				'out'           : out,
				'idx'           : idx,
				'feature'       : feature_desc
			})

		self.launcher.run(ac.compute_all_channel, features=features,
						  caption='Computing Multiple Feature',
						  cb=self.on_all_computed)

	def on_all_computed(self, results):
		for result in results:
			if result is None:
				continue
			out, idx, params = result
			self.DM.channel_computed.emit(out, params)
			self.features[idx].setComputed(True)


	def on_features_computed(self, result):
		if result is None:
			QtGui.QMessageBox.critical(self, 'Error', 'Feature not available')
		else:
			out, idx, params = result
			self.DM.channel_computed.emit(out, params)
			self.features[idx].setComputed(True)
			self.DM.update_channel.emit(out)
