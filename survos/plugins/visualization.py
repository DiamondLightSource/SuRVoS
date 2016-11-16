
import numpy as np
from ..qt_compat import QtGui, QtCore

from .base import Plugin
from .contrast import Contrast
from .layers import Layers
from ..widgets import HWidgets, LCheckBox, HSize3D, BLabel, PLineEdit, HeaderLabel
from ..core import DataModel, LayerManager, Launcher
from .. import actions as ac


class LayersMenu(QtGui.QMenu):

    def __init__(self, parent=None):
        super(LayersMenu, self).__init__(parent=parent)
        self.setStyleSheet('QMenu {'
                           '    border: 1px solid #ccc;'
                           '    background-color: #fbfbfb;'
                           '    border-radius: 4px;'
                           '    min-width: 400px;'
                           '}')
        layout = QtGui.QVBoxLayout(self)
        self.layers = Layers()
        layout.addWidget(self.layers)

    def sizeHint(self):
        return self.layers.sizeHint()

class ContrastMenu(QtGui.QMenu):

    def __init__(self, parent=None):
        super(ContrastMenu, self).__init__(parent=parent)
        self.setStyleSheet('QMenu {'
                           '    border: 1px solid #ccc;'
                           '    background-color: #fbfbfb;'
                           '    border-radius: 4px;'
                           '    min-width: 400px;'
                           '}')
        layout = QtGui.QVBoxLayout(self)
        self.contrast = Contrast()
        layout.addWidget(self.contrast)

    def sizeHint(self):
        return self.contrast.sizeHint()

class Visualization(Plugin):

    name = 'Visualization'

    def __init__(self, parent=None):
        super(Visualization, self).__init__(ptype=Plugin.Plugin, parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()

        contrastLabel = HeaderLabel('Contrast')
        layersLabel = HeaderLabel('Layers')

        self.addWidget(contrastLabel)
        self.addWidget(Contrast())
        self.addWidget(layersLabel)
        self.addWidget(Layers())
