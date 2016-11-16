
import numpy as np
from ..qt_compat import QtGui, QtCore

import logging as log

from matplotlib.colors import ListedColormap

from ..core import Launcher
from ..widgets import HSize3D, HWidgets, PLineEdit, SourceCombo, ActionButton
from .base import Plugin
from ..core import DataModel, LayerManager
from .. import actions as ac

from ..lib._features import find_boundaries

class MegaVoxels(QtGui.QWidget):

    name = 'MegaVoxels'

    def __init__(self, parent=None):
        super(MegaVoxels, self).__init__(parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.setLayout(QtGui.QVBoxLayout())

        self.source_combo = SourceCombo()
        self.layout().addWidget(HWidgets(None, 'Source:', self.source_combo, stretch=[1]))

        self.txt_lambda = PLineEdit(0.1, parse=float)
        self.txt_bins = PLineEdit(20, parse=int)
        dummy = HWidgets(QtGui.QLabel('Lamda:'), self.txt_lambda,
                         QtGui.QLabel('NumBins:'), self.txt_bins,
                         stretch=[1,0,1,0])
        self.layout().addWidget(dummy)

        self.txt_gamma = PLineEdit(None, parse=self.parse_gamma)
        dummy = HWidgets(None, QtGui.QLabel('Gamma:'), self.txt_gamma,
                         stretch=[1,0,0])
        self.layout().addWidget(dummy)

        self.btn_apply = ActionButton('Apply')

        dummy = HWidgets(QtGui.QWidget(), self.btn_apply,\
                         stretch=[1,0,])

        self.layout().addWidget(dummy)

        self.btn_apply.clicked.connect(self.apply_megavoxels)

    def parse_gamma(self, val):
        try:
            val = int(val)
            return val
        except:
            return None

    def apply_megavoxels(self):
        target = self.source_combo.value()
        ds = 'data/{}'.format(target)

        if self.DM.svtotal is None or not self.DM.has_ds('supervoxels/supervoxels'):
            Launcher.instance().error.emit('SuperVoxels need to be computed first')
            return

        self.params = {
            'lamda' : self.txt_lambda.value(),
            'nbins' : self.txt_bins.value(),
            'gamma' : self.txt_gamma.value()
        }

        in_data = target
        in_sv = 'supervoxels/supervoxels'

        out_mv = 'megavoxels/megavoxels'
        out_mvindex = 'megavoxels/megavoxels_idx'
        out_mvtable = 'megavoxels/megavoxels_table'
        self.DM.remove_dataset(out_mv)
        self.DM.remove_dataset(out_mvindex)
        self.DM.remove_dataset(out_mvtable)

        Launcher.instance().run(ac.create_megavoxels,
                                dataset=in_data,
                                splabels=in_sv,
                                num_sps=self.DM.svtotal,
                                out_mv=out_mv,
                                out_mvindex=out_mvindex,
                                out_mvtable=out_mvtable,
                                cb=self.on_megavoxels,
                                caption='Creating MegaVoxels..',
                                **self.params)

    def on_megavoxels(self, params):
        megavoxels, total_mv, sortindex, sorttable = params
        self.update_megavoxel_layer(megavoxels, sortindex, sorttable, total_mv, visible=True)

    def update_megavoxel_layer(self, mvlabels, sortindex, sorttable, mvtotal, visible=False):
        self.DM.mvlabels = mvlabels
        self.DM.mvindex = sortindex
        self.DM.mvtable = sorttable
        self.DM.mvtotal = mvtotal

        if self.LM.isin('MegaVoxels', 'Super-Regions'):
            self.LM.get('MegaVoxels', 'Super-Regions').data = self.DM.mvlabels
            self.LM.setVisible('MegaVoxels', 'Super-Regions', True)
        else:
            cmap = ListedColormap(['Maroon'])
            self.LM.addLayer(self.DM.mvlabels, 'MegaVoxels', level='Super-Regions',
                             pre=find_boundaries, cmap=cmap, background=0,
                             vmin=0.1, vmax=1, visible=visible)
        self.LM.update()

    def update_params(self, source, total_mv, lamda, nbins, gamma):
        log.info('  * {} megavoxels loaded'.format(total_mv))
        self.source_combo.loadSource(source)
        self.txt_lambda.setText(str(lamda))
        self.txt_bins.setText(str(nbins))
        self.txt_gamma.setText(str(gamma))
