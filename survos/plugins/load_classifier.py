
import numpy as np
from ..qt_compat import QtGui, QtCore, QtWidgets

from .base import Plugin
from ..widgets import HWidgets, LCheckBox, HSize3D, BLabel, PLineEdit, HeaderLabel
from ..core import DataModel, LayerManager, Launcher
from .. import actions as ac
from ..widgets import ActionButton



class LoadSaveClassifier(QtWidgets.QWidget):

    load = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(LoadSaveClassifier, self).__init__(parent=parent)

        #self.idx = idx

        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(hbox)

        self.apply = ActionButton('Load')
        hbox.addWidget(self.apply)

        #self.apply.clicked.connect(self.on_load_label)

    #def on_load_label(self):
        #self.load.emit(self.idx)


class LoadClassifier(Plugin):

    name = 'Load Classifier'
    load = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(LoadClassifier, self).__init__(ptype=Plugin.Plugin, parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()

        loadLabel = HeaderLabel('Load Trained Classifier')

        self.addWidget(loadLabel)
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.apply = ActionButton('Load')
        vbox.addWidget(self.apply)
        #self.addWidget(LoadSaveClassifier)

        self.apply.clicked.connect(self.on_load_label)

    def on_load_label(self):
        print("Button Clicked!")
        self.DM.load_classifier()
