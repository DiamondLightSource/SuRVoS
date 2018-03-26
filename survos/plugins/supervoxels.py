
import numpy as np
from ..qt_compat import QtGui, QtCore, QtWidgets

import logging as log

from matplotlib.colors import ListedColormap

from ..core import Launcher
from ..widgets import HSize3D, HWidgets, PLineEdit, SourceCombo, ActionButton
from .base import Plugin
from ..core import DataModel, LayerManager
from .. import actions as ac

from ..lib._features import find_boundaries

class SuperVoxels(QtWidgets.QWidget):

    name = 'SuperVoxels'

    def __init__(self, parent=None):
        super(SuperVoxels, self).__init__(parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.setLayout(QtWidgets.QVBoxLayout())

        self.source_combo = SourceCombo()
        self.layout().addWidget(HWidgets(None, 'Source:', self.source_combo, stretch=[1]))

        self.sp_size = HSize3D('SP shape', (10, 10, 10), txtwidth=70)
        self.sp_spacing = HSize3D('Spacing', (1, 1, 1), txtwidth=70)

        self.layout().addWidget(self.sp_size)
        self.layout().addWidget(self.sp_spacing)

        self.txt_comp = PLineEdit(20., parse=float)

        dummy = HWidgets('Compactness:', self.txt_comp, \
                         stretch=[0,0,0,0])
        self.layout().addWidget(dummy)


        self.btn_apply = ActionButton('Apply')
        dummy = HWidgets(QtWidgets.QWidget(), self.btn_apply,\
                         stretch=[1,0])

        self.layout().addWidget(dummy)

        self.btn_apply.clicked.connect(self.apply_supervoxels)

    def apply_supervoxels(self):
        target = self.source_combo.value()

        spshape = self.sp_size.value()

        self.params = {
            'sp_shape' : spshape,
            'spacing' : self.sp_spacing.value(),
            'compactness' : self.txt_comp.value()
        }

        in_data = target

        out_sv = 'supervoxels/supervoxels'
        out_svindex = 'supervoxels/supervoxels_idx'
        out_svtable = 'supervoxels/supervoxels_table'
        out_svedges = 'supervoxels/graph_edges'
        out_svweights = 'supervoxels/graph_edge_weights'

        self.DM.remove_dataset(out_sv)
        self.DM.remove_dataset(out_svindex)
        self.DM.remove_dataset(out_svtable)
        self.DM.remove_dataset(out_svedges)
        self.DM.remove_dataset(out_svweights)

        Launcher.instance().run(ac.create_supervoxels,
                                dataset=in_data,
                                out_sv=out_sv,
                                out_svindex=out_svindex, out_svtable=out_svtable,
                                out_svedges=out_svedges, out_svweights=out_svweights,
                                cb=self.on_supervoxels,
                                caption='Creating SuperVoxels..',
                                **self.params)

    def on_supervoxels(self, params):
        svlabels, svtotal, sortindex, sorttable, edges, weights = params
        self.update_supervoxel_layer(svlabels, sortindex, sorttable, svtotal, visible=True)

    def update_supervoxel_layer(self, svlabels, sortindex, sorttable, total_sv, visible=False):
        if svlabels is not None:
            self.DM.svlabels = svlabels
            self.DM.svtotal = total_sv
            self.DM.svindex = sortindex
            self.DM.svtable = sorttable

        if self.LM.isin('SuperVoxels', 'Super-Regions'):
            self.LM.get('SuperVoxels', 'Super-Regions').data = self.DM.svlabels
            self.LM.setVisible('SuperVoxels', 'Super-Regions', True)
        else:
            cmap = ListedColormap(['MidnightBlue'])
            self.LM.addLayer(self.DM.svlabels, 'SuperVoxels', level='Super-Regions',
                             pre=find_boundaries, cmap=cmap, background=0,
                             vmin=0.1, vmax=1, visible=visible)
        self.LM.update()

    def update_params(self, source, total_sv, sv_shape, svacing, compactness):
        log.info('  * {} supervoxels loaded'.format(total_sv))
        self.source_combo.loadSource(source)
        self.sp_size.setValue(sv_shape)
        self.sp_spacing.setValue(svacing)
        self.txt_comp.setText(str(compactness))
