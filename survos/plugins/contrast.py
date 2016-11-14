
import numpy as np
from ..qt_compat import QtGui, QtCore

from ..widgets import HSlider, HWidgets, ActionButton
from ..widgets.mpl_widgets import MplCanvas
from .base import Plugin
from ..core import DataModel, LayerManager, Launcher

import logging as log
import seaborn as sns
from matplotlib import pyplot as plt

class Contrast(QtGui.QWidget):

	name = 'Contrast'

	def __init__(self, parent=None):
		super(Contrast, self).__init__(parent=parent)

		vbox = QtGui.QVBoxLayout(self)
		self.setLayout(vbox)

		self.DM = DataModel.instance()
		self.LM = LayerManager.instance()
		self.launcher = Launcher.instance()

		self.sld_vmin = HSlider('VMin', 0, 1, 0)
		self.sld_vmax = HSlider('VMax', 0, 1, 1)

		vbox.addWidget(self.sld_vmin)
		vbox.addWidget(self.sld_vmax)

		visualize = QtGui.QCheckBox('View Histogram')
		default = ActionButton('Default')
		default.clicked.connect(self.restore_default)
		vbox.addWidget(HWidgets(visualize, None, default, stretch=[0,1,0]))

		self.visualization = MplCanvas(autoscale=True, axisoff=True)
		self.visualization.setFixedWidth(293)
		self.visualization.setFixedHeight(200)
		self.visualization.setVisible(False)
		visualize.toggled.connect(self.on_visualize)
		vbox.addWidget(HWidgets(None, self.visualization, None, stretch=[1,0,1]))

		# Slots
		self.DM.vmin_changed.connect(self.update_vmin)
		self.DM.vmax_changed.connect(self.update_vmax)
		self.DM.evmin_changed.connect(self.update_evmin)
		self.DM.evmax_changed.connect(self.update_evmax)

		self.sld_vmin.valueChanged.connect(self.evmin_changed)
		self.sld_vmax.valueChanged.connect(self.evmax_changed)

		self.DM.select_channel.connect(self.view_channel)
		self.DM.update_channel.connect(self.view_channel)

		self.hist_computed = False
		self.current_channel = '/data'
		self.vmin = self.vmax = self.evmin = self.evmax = 0
		self.update_contrast(self.current_channel)

	def view_channel(self, channel='/data'):
		layer = self.LM.get('Data', 'Data')
		layer.data = channel
		vmin, vmax = self.update_contrast(channel)
		layer.vmin = vmin
		layer.vmax = vmax
		self.LM.update()
		if self.visualization.isVisible():
			self.compute_hist()
		else:
			self.hist_computed = False

	def restore_default(self):
		attrs = self.DM.attrs(self.current_channel)
		evmin = attrs['default_evmin']
		evmax = attrs['default_evmax']
		self.evmin_changed(evmin)
		self.evmax_changed(evmax)
		self.LM.update()

	def update_contrast(self, channel):
		self.current_channel = channel

		attrs = self.DM.attrs(channel)
		vmin = attrs['vmin']
		vmax = attrs['vmax']
		evmin = attrs['evmin']
		evmax = attrs['evmax']

		self.update_vmin(vmin)
		self.update_vmax(vmax)
		self.update_evmin(evmin)
		self.update_evmax(evmax)

		return evmin, evmax

	def evmin_changed(self, val):
		if val >= self.evmax:
			val = self.evmax - 1e-10
		self.DM.set_attrs(self.current_channel, dict(evmin=val))
		self.DM.evmin_changed.emit(val)
		self.LM.setVMin('Data', 'Data', val)

	def evmax_changed(self, val):
		if val <= self.evmin:
			val = self.evmin +  1e-10
		self.DM.set_attrs(self.current_channel, dict(evmax=val))
		self.DM.evmax_changed.emit(val)
		self.LM.setVMax('Data', 'Data', val)

	def update_vmin(self, val):
		self.vmin = val
		self.sld_vmin.setMinimum(val)
		self.sld_vmax.setMinimum(val)

	def update_vmax(self, val):
		self.vmax = val
		self.sld_vmin.setMaximum(val)
		self.sld_vmax.setMaximum(val)

	def update_evmin(self, val):
		self.evmin = val
		if self.hist_computed:
			aux = ([val, val], [0, 1])
			self.minline.set_data(aux)
			aux = self.minspan.get_xy(); aux[2:4, 0] = val
			self.minspan.set_xy(aux)
			self.visualization.redraw()
		self.sld_vmin.blockSignals(True)
		self.sld_vmin.setValue(val)
		self.sld_vmin.blockSignals(False)

	def update_evmax(self, val):
		self.evmax = val
		if self.hist_computed:
			aux = ([val, val], [0, 1])
			self.maxline.set_data(aux)
			aux = self.maxspan.get_xy(); aux[[0,1,4], 0] = val
			self.maxspan.set_xy(aux)
			self.visualization.redraw()
		self.sld_vmax.blockSignals(True)
		self.sld_vmax.setValue(val)
		self.sld_vmax.blockSignals(False)

	def on_visualize(self, bol):
		if bol and not self.hist_computed:
			self.compute_hist()
		self.visualization.setVisible(bol)

	def compute_hist(self):
		self.launcher.setup('Computing histogram')
		x, y = self.DM.get_histogram(self.current_channel)

		log.info('+ Plotting')
		ax = self.visualization.ax
		ax.clear()

		y = np.maximum(0, y)
		width = 0.7 * (y[1] - y[0])
		ax.plot(x, y)
		ax.fill_betweenx(y, 1e-12, x, alpha=0.25)

		self.minspan = ax.axvspan(self.vmin, self.evmin, color='r', alpha=0.2)
		self.minline = ax.axvline(self.evmin, linewidth=1, color='r')

		self.maxspan = ax.axvspan(self.evmax, self.vmax, color='r', alpha=0.2)
		self.maxline = ax.axvline(self.evmax, linewidth=1, color='r')

		ax.set_xlim([self.vmin, self.vmax])
		ax.set_xticks([])
		ax.set_yticks([])

		self.visualization.redraw()

		self.hist_computed = True
		self.launcher.cleanup()
