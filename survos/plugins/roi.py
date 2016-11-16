
import numpy as np
from ..qt_compat import QtGui, QtCore

import logging as log

from collections import OrderedDict

from .base import Plugin
from ..widgets import HSize3D, HeaderLabel, SubHeaderLabel, \
                      HWidgets, RoundedWidget, ActionButton
from ..core import DataModel, Launcher, LayerManager
from .. import actions as ac


class ROIWidget(RoundedWidget):

    selected = QtCore.pyqtSignal(int)
    removed = QtCore.pyqtSignal(int)

    def __init__(self, idx, dfrom, dto, checked=False, removable=True, parent=None):
        super(ROIWidget, self).__init__(parent=parent, color=None, bg='#cde5e5',
                                        width=0)
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)

        self.idx = idx
        self.dfrom = dfrom
        self.dto = dto

        if removable:
            self.btn_rm = QtGui.QPushButton('X')
            self.btn_rm.clicked.connect(self.on_remove)
        else:
            self.btn_rm = QtGui.QWidget()
        self.btn_rm.setMinimumWidth(25)

        self.labelz = QtGui.QLabel('[{}, {})'.format(dfrom[0], dto[0]))
        self.labelz.setFixedWidth(67)
        self.labelz.setStyleSheet('color: #009999;')
        self.labely = QtGui.QLabel('[{}, {})'.format(dfrom[1], dto[1]))
        self.labely.setFixedWidth(67)
        self.labely.setStyleSheet('color: #009999;')
        self.labelx = QtGui.QLabel('[{}, {})'.format(dfrom[2], dto[2]))
        self.labelx.setFixedWidth(67)
        self.labelx.setStyleSheet('color: #009999;')

        self.z = QtGui.QLabel('z:')
        self.z.setStyleSheet('color: #6194BC; font-weight: bold;')
        self.z.setFixedWidth(7)
        self.y = QtGui.QLabel('y:')
        self.y.setStyleSheet('color: #6194BC; font-weight: bold;')
        self.y.setFixedWidth(7)
        self.x = QtGui.QLabel('x:')
        self.x.setStyleSheet('color: #6194BC; font-weight: bold;')
        self.x.setFixedWidth(7)

        self.chk = QtGui.QCheckBox()
        self.chk.setChecked(checked)
        self.chk.stateChanged.connect(self.mousePressEvent)

        vbox.addWidget(HWidgets(self.btn_rm, self.z, self.labelz,
                                self.y, self.labely, self.x, self.labelx,
                                None, self.chk, stretch=[0,0,0,0,0,0,0,1,0]))

    def mousePressEvent(self, ev):
        self.setChecked(~self.isChecked())
        self.selected.emit(self.idx)

    def setChecked(self, value):
        self.chk.blockSignals(True)
        self.chk.setChecked(value)
        self.chk.blockSignals(False)

    def isChecked(self):
        return self.chk.isChecked()

    def on_selected(self):
        self.selected.emit(self.idx)

    def value(self):
        slice_z = slice(self.dfrom[0], self.dto[0])
        slice_y = slice(self.dfrom[1], self.dto[1])
        slice_x = slice(self.dfrom[2], self.dto[2])
        return slice_z, slice_y, slice_x

    def on_remove(self):
        self.removed.emit(self.idx)


class ROI(Plugin):

    name = 'Select ROI'

    def __init__(self, parent=None):
        super(ROI, self).__init__(ptype=Plugin.Plugin, parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()

        self.addWidget(HeaderLabel('Select Region of Interest'))

        dummy = QtGui.QWidget(self)
        self.addWidget(dummy)

        vbox = QtGui.QVBoxLayout(self)
        dummy.setLayout(vbox)

        shape = self.DM.data_shape

        self.lbl_size = QtGui.QLabel('<b>Shape:</b> ' + str(shape))

        self.txt_from = HSize3D('From', default=(0, 0, 0), txtwidth=50)
        self.txt_to = HSize3D('To', default=shape, txtwidth=50)

        vbox.addWidget(self.lbl_size)
        vbox.addWidget(self.txt_from)
        vbox.addWidget(self.txt_to)

        self.btn_add = ActionButton('Add')
        vbox.addWidget(HWidgets(None, self.btn_add, stretch=[1,0]))

        vbox.addWidget(SubHeaderLabel('Available ROI'))

        self.container = QtGui.QWidget(self)
        self.container.setLayout(QtGui.QVBoxLayout(self))
        vbox.addWidget(self.container)

        self.rois = OrderedDict()
        self.nrois = 0
        self.addROI((0,0,0), shape, checked=True, removable=False)

        self.btn_add.clicked.connect(self.on_add_roi)

    def on_add_roi(self):
        dfrom = self.txt_from.value()
        dto = self.txt_to.value()
        self.addROI(dfrom, dto)
        self.saveROIs()

    def addROI(self, dfrom, dto, checked=False, removable=True, *args):
        idx = self.nrois
        roi = ROIWidget(idx, dfrom, dto, checked=checked, removable=removable)
        roi.selected.connect(self.on_select)
        roi.removed.connect(self.on_remove)

        self.rois[self.nrois] = roi
        self.container.layout().addWidget(roi)

        self.nrois += 1
        return idx

    def on_remove(self, idx):
        if self.rois[idx].isChecked():
            self.on_select(0)
        self.rois[idx].setParent(None)
        del self.rois[idx]
        self.saveROIs()

    def on_select(self, idx):
        for k, v in self.rois.items():
            if k != idx:
                v.setChecked(False)

        if not self.rois[idx].isChecked():
            self.rois[idx].setChecked(False)
            self.rois[0].setChecked(True)
            idx = 0

        self.DM.active_roi = self.rois[idx].value()
        self.DM.roi_changed.emit()
        self.LM.update()
        self.saveROIs()

    def saveROIs(self):
        rois = []
        for k, v in self.rois.items():
            if k == 0: continue
            roi = v.value()
            dfrom = [r.start for r in roi]
            dto = [r.stop for r in roi]
            rois.append(dfrom + dto + [int(v.isChecked())])
        self.DM.set_attrs('/data', dict(roi=rois))

    def loadROIs(self):
        rois = self.DM.attr('/data', 'roi')
        if rois is None:
            return
        selected = 0
        for roi in rois:
            dfrom = tuple(roi[:3])
            dto = tuple(roi[3:6])
            checked = roi[-1] == 1
            i = self.addROI(dfrom, dto, checked=checked)
            if checked:
                selected = i
        self.on_select(selected)
