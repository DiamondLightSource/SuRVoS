
import numpy as np
from ..qt_compat import QtGui, QtCore
from collections import OrderedDict

from .model import DataModel
from .singleton import Singleton

from matplotlib import cm

Axial = DataModel.instance().Axial
Sagittal = DataModel.instance().Sagittal
Coronal = DataModel.instance().Coronal


class Layer(object):

	def __init__(self, data, level=None, cmap=None, vmin=None, vmax=None, alpha=1., \
				 orient=Axial, background=None, threshold=None, pre=None, visible=True):
		self.DM = DataModel.instance()
		self.data = data
		self.level = level
		self.orient = orient
		self.background = background
		self.cmap = cmap
		self.vmin = vmin
		self.vmax = vmax
		self.pre = pre
		self.alpha = alpha
		self.visible = visible
		self.index = -1
		self.threshold = threshold

	def draw(self, ax, idx, i):
		self.index = i
		if self.visible:
			sslice = self.get_slice(self.data, idx)
			if self.pre is not None:
				sslice = self.pre(sslice)
			if self.background is not None and self.cmap is not None:
				sslice = np.ma.masked_equal(sslice, self.background)
			elif self.threshold is not None and self.cmap is not None:
				target, thresh = self.threshold
				sslice = np.ma.masked_where(self.get_slice(target, idx) < thresh, sslice)
			ax.imshow(sslice, self.cmap, vmin=self.vmin, vmax=self.vmax,
					  alpha=self.alpha, interpolation='none')

	def update(self, image, idx):
		if self.visible:
			sslice = self.get_slice(self.data, idx)
			if self.pre is not None:
				sslice = self.pre(sslice)
			if self.background is not None and self.cmap is not None:
				sslice = np.ma.masked_equal(sslice, self.background)
			elif self.threshold is not None and self.cmap is not None:
				target, thresh = self.threshold
				sslice = np.ma.masked_where(self.get_slice(target, idx) < thresh, sslice)
		else:
			mask = np.ones(self.shape(), np.bool)
			sslice = np.ma.masked_where(mask, mask)
		image.set_array(sslice)

	def shape(self):
		shape = list(self.data.shape)
		del shape[self.orient]
		return shape

	def get_slice(self, data, idx):
		if type(data) in [str, unicode]:
			return self.DM.load_slices(data, idx)[0]
		else:
			raise Exception('This shouldnt happen')


@Singleton
class LayerManager(QtCore.QObject):

	level_added = QtCore.pyqtSignal(str)
	added = QtCore.pyqtSignal(str, str)
	removed = QtCore.pyqtSignal(str, str)
	updated = QtCore.pyqtSignal()
	opacity = QtCore.pyqtSignal(str, str, float)
	toggled = QtCore.pyqtSignal(str, str, bool)

	def __init__(self):
		QtCore.QObject.__init__(self)
		self._levels = OrderedDict()

	def len(self):
		return len(self._layers)

	def levels(self):
		return self._levels.keys()

	def layers(self, level=None):
		if level is None:
			return [val for lvl in self._levels.keys() for val in self._levels[lvl].values()]
		else:
			return [val for val in self._levels[level].values()]

	def layer_names(self, level=None):
		if level is None:
			return [val for lvl in self._levels.keys() for val in self._levels[lvl].keys()]
		else:
			return [val for val in self._levels[level].keys()]

	def visible_layers(self, level=None):
		if level is None:
			return [val for lvl in self._levels.keys() for val in self._levels[lvl].values() if val.visible]
		else:
			return [val for val in self._levels[level].values() if val.visible]

	def addLayer(self, data, name, level, **kwargs):
		if not level in self._levels:
			self._levels[level] = OrderedDict()
			self.level_added.emit(level)
		if not name in self._levels[level]:
			self._levels[level][name] = Layer(data, level, **kwargs)
			self.added.emit(name, level)
		else:
			self._levels[level][name] = Layer(data, level, **kwargs)

	def get(self, name, level):
		return self._levels[level][name]

	def remove(self, name, level):
		if name in self._levels[level]:
			del self._levels[level][name]
			self.removed.emit(name, level)

	def isin(self, name, level):
		return level in self._levels and str(name) in self._levels[level]

	def index(self, name, level):
		return self._levels[level][name].index if self._levels[level][name].visible else -1

	def clear(self, level=None):
		if level is None:
			self._levels.clear()
		else:
			self._levels[level].clear()

	def setOpacity(self, name, level, value):
		self._levels[level][name].alpha = value
		self.opacity.emit(name, level, value)

	def setVisible(self, name, level, bol):
		self._levels[level][name].visible = bol
		self.toggled.emit(name, level, bol)

	def setVMin(self, name, level, val):
		self._levels[level][name].vmin = val

	def setVMax(self, name, level, val):
		self._levels[level][name].vmax = val

	def update(self):
		self.updated.emit()
