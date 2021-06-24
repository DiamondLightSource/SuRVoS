
import numpy as np
from ..qt_compat import QtGui, QtCore

from .base import Plugin
from .supervoxels import SuperVoxels
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

        self.supervoxels = SuperVoxels()

        self.addWidget(svLabel)
        self.addWidget(self.supervoxels)

    def load_supervoxels(self, data, data_idx, data_table,
                         source, sv_shape, spacing, compactness, svtotal):
        self.supervoxels.update_supervoxel_layer(data, data_idx, data_table, svtotal)
        self.supervoxels.update_params(source, svtotal, sv_shape, spacing, compactness)
