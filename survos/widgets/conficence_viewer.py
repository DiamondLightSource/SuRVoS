
from ..qt_compat import QtGui, QtCore

import numpy as np
import logging as log

import os
import matplotlib

from ..plugins import Plugin
from ..core import DataModel, LayerManager, LabelManager, Launcher
from .mpl_widgets import PerspectiveCanvas
from .base import SComboBox

class ConfidenceViewer(Plugin):

	name = 'Confidence Viewer'

	def __init__(self, ptype=Plugin.Widget):
		super(ConfidenceViewer, self).__init__(ptype=ptype)
		self.DM = DataModel.instance()
		self.LBLM = LabelManager.instance()
		self.DM.confidence_changed.connect(self.replot)
		vbox = QtGui.QVBoxLayout()
		self.layout.addLayout(vbox, 0, 0)

		self.orient = 0
		self.canvases = (None,)
		self.idx = (self.DM.data.shape[self.orient]) // 2

		topbox = QtGui.QHBoxLayout()
		vbox.addLayout(topbox, 0)
		# Perspective
		self.combo = SComboBox()
		self.combo.addItem("Axial")
		self.combo.addItem("Sagittal")
		self.combo.addItem("Coronal")
		self.combo.setCurrentIndex(0)
		self.combo.currentIndexChanged.connect(self.on_combo)
		topbox.addWidget(self.combo)

		# Slider
		self.slider = QtGui.QSlider(1)
		self.slider.setMinimum(0)
		self.slider.setMaximum(self.DM.data.shape[0]-1)
		self.slider.setValue(self.idx)
		self.slider.valueChanged.connect(self.on_value)
		self.slider.setSingleStep(1)
		topbox.addWidget(self.slider)

		# Slider Text
		self.text_idx = QtGui.QLabel(str(self.idx))
		topbox.addWidget(self.text_idx)

		self.container = QtGui.QWidget()
		self.container.setLayout(QtGui.QGridLayout())
		vbox.addWidget(self.container, 1)

	def on_value(self, idx):
		self.idx = idx
		self.text_idx.setText(str(self.idx))

	def clear(self):
		for widget in self.canvases:
			if widget is not None:
				widget.setParent(None)
				widget.deleteLater()
		self.canvases = (None,)

	def on_combo(self, orient):
		self.orient = orient
		self.idx = (self.DM.data.shape[self.orient]) // 2
		for widget in self.canvases:
			if widget is not None:
				widget.idx = self.idx
				widget.orient = orient
				widget.replot()

	def replot(self):
		self.clear()
		total = self.LBLM.len()
		spunique = np.unique(self.DM.gtlabels[:])
		cols = total if total < 4 else total//2 if total%2==0 else total//2+1
		rows = total//cols if total%cols==0 else total//cols+1
		k = 0
		labels = list(self.LBLM.labels())
		self.canvases = []
		for i in range(rows):
			for j in range(cols):
				name = None if i*cols+j >= len(labels) else labels[i*cols+j].name
				if k in spunique:
					widget = ConfidenceCanvas(k, name=name)
					k += 1
				else:
					widget = ConfidenceCanvas(name=name)
				self.canvases += [widget]
				self.slider.valueChanged.connect(widget.update_volume)
				self.container.layout().addWidget(widget, i, j)
