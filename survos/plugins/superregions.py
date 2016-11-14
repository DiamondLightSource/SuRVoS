
import numpy as np
from ..qt_compat import QtGui, QtCore

from .base import Plugin
from .supervoxels import SuperVoxels
from .megavoxels import MegaVoxels
from ..widgets import HWidgets, LCheckBox, HSize3D, BLabel, PLineEdit, HeaderLabel
from ..core import DataModel, LayerManager, Launcher
from .. import actions as ac


class SuperRegions(Plugin):

	name = 'Super Regions'

	def __init__(self):
		super(SuperRegions, self).__init__(ptype=Plugin.Plugin)

		self.DM = DataModel.instance()
		self.LM = LayerManager.instance()

		svLabel = HeaderLabel('SuperVoxels')
		mvLabel = HeaderLabel('MegaVoxels')

		self.supervoxels = SuperVoxels()
		self.megavoxels = MegaVoxels()

		self.addWidget(svLabel)
		self.addWidget(self.supervoxels)
		self.addWidget(mvLabel)
		self.addWidget(self.megavoxels)

	def load_supervoxels(self, data, data_idx, data_table,
						 source, sv_shape, spacing, compactness, svtotal):
		self.supervoxels.update_supervoxel_layer(data, data_idx, data_table, svtotal)
		self.supervoxels.update_params(source, svtotal, sv_shape, spacing, compactness)

	def load_megavoxels(self, data, data_idx, data_table,
						source, lamda, nbins, gamma, mvtotal):
		self.megavoxels.update_megavoxel_layer(data, data_idx, data_table, mvtotal)
		self.megavoxels.update_params(source, mvtotal, lamda, nbins, gamma)
