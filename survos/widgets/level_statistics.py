

import numpy as np
import pandas as pd

from ..qt_compat import QtGui, QtCore

from .mpl_widgets import PerspectiveCanvas, MplCanvas
from .base import HWidgets, TComboBox, RoundedWidget, ColorButton, PLineEdit, \
				  LCheckBox, SComboBox, CheckableLabels, SourceCombo

from .. import actions as ac

from ..plugins.base import Plugin
from ..core import DataModel, LabelManager, Launcher, LayerManager


import logging as log
from collections import OrderedDict

import seaborn as sns

from .label_partitioning import FEATURE_TYPES, FEATURE_OPTIONS


class LevelStats(Plugin):

	name = 'Level Statistics'

	def __init__(self, ptype=Plugin.Widget):
		super(LevelStats, self).__init__(ptype=ptype)

		self.DM = DataModel.instance()
		self.LBLM = LabelManager.instance()

		self.level_combo = SComboBox()
		self.level_combo.setMinimumWidth(200)
		self.label_combo = CheckableLabels(-1)
		self.label_combo.setMinimumWidth(200)
		self.source_combo = SourceCombo()
		self.source_combo.setMinimumWidth(200)
		self.label_btn = QtGui.QPushButton('Label')
		self.addWidget(HWidgets(self.level_combo, self.label_combo,
								self.source_combo, self.label_btn,
								stretch=[1, 1, 1, 1]))

		self.mplcanvas = MplCanvas(axisoff=False, autoscale=True)
		self.mplcanvas.ax.set_yticks([])

		self.kernels = TComboBox('Fit kernel:', ['gau', 'cos', 'biw', 'epa',
												'tri', 'triw'],
								 selected=0)
		self.update_plot = QtGui.QPushButton('Update plot')
		self.export_plot = QtGui.QPushButton('Export plot')
		self.export_stats = QtGui.QPushButton('Export Stats')
		self.addWidget(HWidgets(self.kernels, None, self.update_plot,
								self.export_plot, self.export_stats,
								stretch=[0, 1, 0, 0]))

		splitter = QtGui.QSplitter(1)
		self.addWidget(splitter)

		group = QtGui.QGroupBox()
		vbox = QtGui.QVBoxLayout()
		group.setLayout(vbox)

		self.options = OrderedDict()

		for name, define in zip(FEATURE_OPTIONS, FEATURE_TYPES):
			dummy = LCheckBox(name)
			dummy.setChecked(False)
			self.options[define] = (name, dummy)
			vbox.addWidget(dummy)

		splitter.addWidget(self.mplcanvas)
		splitter.addWidget(group)

		self.LBLM.levelLoaded.connect(self.on_level_added)
		self.LBLM.levelAdded.connect(self.on_level_added)
		self.LBLM.levelRemoved.connect(self.on_level_removed)
		self.level_combo.currentIndexChanged.connect(self.on_level_changed)
		self.label_btn.clicked.connect(self.on_label_clicked)

		self.update_plot.clicked.connect(self.on_update_plot)
		self.export_plot.clicked.connect(self.on_export_plot)
		self.export_stats.clicked.connect(self.on_export_stats)
		self.kernels.currentIndexChanged.connect(self.on_kernel_changed)

		self.levels = []
		self.selected_label = None
		self.kernel = 'gau'
		self.feature = None
		self.colors = None
		self.color_idx = None
		self.labels = None
		self.objects = None
		self.computed_level = None

	def on_level_changed(self, index):
		self.label_combo.clear()

		if index < 0:
			return

		self.label_combo.selectLevel(self.levels[index])

	def on_level_added(self, level):
		self.levels.append(level)
		self.level_combo.addItem('Level {}'.format(level))
		if level == 0:
			self.on_level_changed(0)
			self.label_combo.selectLevel(level)

	def on_level_removed(self, level):
		idx = self.levels.index(level)
		self.level_combo.removeItem(idx)
		if self.computed_level is not None and level == self.computed_level:
			self.reset()

	def on_label_clicked(self):
		self.reset()

		source = self.source_combo.value()
		level = self.levels[self.level_combo.currentIndex()]
		self.computed_level = level
		self.LBLM.save(level)

		labels = self.label_combo.getSelectedLabels()

		out_features = []
		for f in FEATURE_TYPES:
			out_features.append('objects/{}'.format(f))

		in_dset = self.LBLM.dataset(level)
		out_dset = 'objects/objects'
		self.DM.remove_dataset(out_dset)

		Launcher.instance().run(ac.label_objects, dataset=in_dset, source=source,
								out=out_dset, out_features=out_features,
								labels=labels, caption='Labelling objects...',
								return_labels=True, cb=self.on_objects_labelled)

	def on_objects_labelled(self, params):
		self.objects, self.num_objects, self.labels = params
		if self.num_objects == 0:
			QtGui.QMessageBox.critical(self, "Error", 'No objects found')
			return
		self.on_update_plot()


	def on_update_plot(self, res=None):
		if self.objects is None:
			QtGui.QMessageBox.critical(self, 'Error', 'No objects labelled')
			return

		self.replot()

	def replot(self):
		self.mplcanvas.fig.clear()
		self.mplcanvas.ax.clear()
		colors = {int(l.idx) : l.color for l in self.LBLM.labels(self.computed_level)}

		selected = []
		selected_title = []
		for d, v in self.options.items():
			if v[1].isChecked():
				selected += [d]
				selected_title += [v[0]]

		N = len(selected)
		axes = np.empty((N, N), dtype=object)

		with sns.axes_style('ticks'):
			for i in range(N):
				ds_i = selected[i]
				data_i = self.DM.load_ds('objects/{}'.format(ds_i))
				for j in range(N-1, -1, -1):
					ds_j = selected[j]
					data_j = self.DM.load_ds('objects/{}'.format(ds_j))
					sharex = axes[-1, j]
					sharey = None
					if i != j:
						sharey = axes[i, 0] if i > 0 else axes[i, 1]

					axes[i, j] = ax = self.mplcanvas.fig.add_subplot(N, N, i*N+j+1,
																	 sharey=sharey,
																	 sharex=sharex)
					if i == j:
						for l in np.unique(self.labels):
							sns.kdeplot(data_i[self.labels == l], shade=True, ax=ax,
										color=colors[l-1], kernel=self.kernel)
						ax.yaxis.set_ticklabels([])
					else:
						c = [colors[int(l-1)] for l in self.labels]
						ax.scatter(data_j, data_i, c=c, linewidths=1,
								   edgecolor="w", s=40)

					if i < N - 1:
						ax.set_xticklabels(ax.get_xticklabels(), visible=False)

			for i in range(N):
				axes[-1, i].set_xlabel(selected_title[i])
				axes[i, 0].set_ylabel(selected_title[i])


			sns.despine(self.mplcanvas.fig)
			self.mplcanvas.redraw()

	def on_export_plot(self):
		full_path = QtGui.QFileDialog.getSaveFileName(self, "Select output filename",
													  filter='*.png')
		if full_path is not None and len(full_path) > 0:
			if not full_path.endswith('.png'):
				full_path += '.png'
		self.mplcanvas.fig.savefig(full_path)

	def on_export_stats(self):
		features = FEATURE_TYPES

		total = len(features)
		nobj = self.num_objects
		data = np.zeros((nobj, total), np.float32)

		for n, ftype in enumerate(features):
			feature = self.DM.load_ds('objects/{}'.format(ftype))
			data[:, n] = feature

		full_path = QtGui.QFileDialog.getSaveFileName(self, "Select output filename",
													  filter='*.csv')
		if full_path is not None and len(full_path) > 0:
			if not full_path.endswith('.csv'):
				full_path += '.csv'
			df = pd.DataFrame(data, columns=features)
			if self.labels is not None and len(self.labels) > 0:
				df['class'] = [self.LBLM.get(self.computed_level, l-1).name for l in self.labels]
			df.to_csv(full_path)

	def on_kernel_changed(self):
		self.kernel = self.kernels.currentText()
		self.replot()

	def reset(self):
		for k, v in self.options.iteritems():
			if type(k) == tuple:
				v[1].setParent(None)
				del self.options[k]

		self.objects = None
		self.selected_label = None
		self.feature = None
		self.colors = None
		self.color_idx = None
		self.labels = None
		self.computed_level = None
		self.mplcanvas.ax.clear()
		self.mplcanvas.redraw()
