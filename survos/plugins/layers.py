
import numpy as np
from ..qt_compat import QtGui, QtCore

from collections import OrderedDict

from .base import Plugin
from ..widgets import HLayer, HeaderLabel, SubHeaderLabel
from ..core import DataModel, LayerManager, Launcher
from .. import actions as ac


class Layers(QtGui.QWidget):

    name = 'Layers'

    def __init__(self, parent=None):
        super(Layers, self).__init__(parent=parent)

        self.setLayout(QtGui.QVBoxLayout())

        self.levels = OrderedDict()
        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LM.added.connect(self.on_layer_added)
        self.LM.removed.connect(self.on_layer_removed)
        self.LM.toggled.connect(self.setVisibility)

    def __del__(self):
        self.levels.clear()

    def on_layer_added(self, name, level):
        name, level = str(name), str(level)
        if level not in self.levels:
            self.levels[level] = OrderedDict()

        lobj = self.LM.get(name, level)
        layer = HLayer(name, level, current=int(lobj.alpha*100), visible=lobj.visible)
        layer.valueChanged.connect(self.on_value)
        layer.toggled.connect(self.on_toggled)
        layer.export.connect(self.on_tiff_export)
        self.levels[level][name] = layer

        self.redraw_layers()

    def redraw_layers(self):
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)

        for level in self.LM.levels():
            header = SubHeaderLabel(level)
            self.layout().addWidget(header)
            for layer in self.LM.layer_names(level=level):
                self.layout().addWidget(self.levels[level][layer])
        self.update()

    def on_layer_removed(self, name, level):
        del self.levels[level][name]
        self.redraw_layers()

    def on_value(self, name, level, val):
        self.LM.setOpacity(name, level, val / 100.)

    def on_toggled(self, name, level, bol):
        self.LM.setVisible(name, level, bol)
        self.LM.update()

    def setVisibility(self, name, level, bol):
        self.levels[level][name].setChecked(bol)

    def on_tiff_export(self, name, level):
        path = str(QtGui.QFileDialog.getSaveFileName(self, filter='*.rec'))
        if path is None or path == '':
            return
        if not path.endswith('.rec'):
            path += '.rec'
        layer = self.LM.get(name, level)
        Launcher.instance().run(ac.export_tiff_layer, data=layer.data, level=level,
                                path=path, cb=None, caption='Exporting layer %s..' % name)
