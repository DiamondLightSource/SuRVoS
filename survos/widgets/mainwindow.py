
import math
import numpy as np

from ..qt_compat import QtGui, QtCore

import logging as log
import time
import re
import shutil

from .slice_viewer import SliceViewer
from .label_partitioning import LabelSplitter
from .level_statistics import LevelStats
from .preloader import PreWidget

from ..core import Launcher
from ..core import DataModel, LayerManager, LabelManager
from ..plugins import Plugin
from ..plugins import ROI, SuperRegions, Annotations, Visualization,\
					  Training, FeatureChannels, Export
from ..widgets import RCheckBox, HeaderButton, HWidgets, BLabel, PicButton, \
					  ComboDialog

from time import sleep

from .. import actions as ac

import os
import fnmatch
import h5py as h5

class QPlainTextEditLogger(log.Handler):
	def __init__(self, parent):
		super(QPlainTextEditLogger, self).__init__()
		self.widget = parent

	def emit(self, record):
		self.widget.setText(record.message)

	def write(self, m):
		pass

class MainWindow(QtGui.QMainWindow):

	def __init__(self, title='QMainWindow'):
		super(MainWindow, self).__init__()

		self.DM = None

		self.pre_widget = PreWidget()
		self.main_widget = MainWidget()
		self.main_container = QtGui.QStackedWidget()
		self.main_container.addWidget(self.pre_widget)
		self.main_container.addWidget(self.main_widget)
		self.setCentralWidget(self.main_container)

		self.setWindowTitle(title)

		self.overlay = Overlay(parent=self)
		self.launcher = Launcher.instance()
		self.launcher.pre.connect(self.overlay.pre)
		self.launcher.post.connect(self.overlay.post)
		self.launcher.error.connect(self.on_error)

		self.status = self.statusBar()
		self.edit = QtGui.QLabel()
		self.edit.setStyleSheet('color: #fefefe; padding-left: 10px;');
		self.edit.setFixedHeight(25)
		self.status.insertPermanentWidget(0, self.edit, 1)

		self.loghandler = QPlainTextEditLogger(self.edit)
		log.getLogger().addHandler(self.loghandler)

		menubar = QtGui.QMenuBar()
		menubar.setStyleSheet('QMenuBar {padding: 6px;}')
		menubar.setContentsMargins(20,10,0,0)
		self.setMenuBar(menubar)
		fileMenu = menubar.addMenu('&File')
		helpMenu = menubar.addMenu('&Help')

		self.openAction = QtGui.QAction('&Open Dataset..', self)
		self.openAction.setShortcut('Ctrl+O')
		self.openAction.setStatusTip('Open Dataset (.rec, .hdf5, .tiff)')
		self.openAction.triggered.connect(self.load_data)
		self.pre_widget.open.clicked.connect(self.load_data)

		self.loadAction = QtGui.QAction('&Load Workspace..', self)
		self.loadAction.setShortcut('Ctrl+L')
		self.loadAction.setStatusTip('Load previously created workspace.')
		self.loadAction.triggered.connect(self.load_workspace)
		self.pre_widget.load.clicked.connect(self.load_workspace)

		self.saveAction = QtGui.QAction('&Save annotations', self)
		self.saveAction.setShortcut('Ctrl+S')
		self.saveAction.setStatusTip('Save and backup current annotations.')
		self.saveAction.triggered.connect(self.main_widget.save_backup)
		self.saveAction.setEnabled(False)

		exitAction = QtGui.QAction('&Exit', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.setStatusTip('Exit application')
		exitAction.triggered.connect(QtGui.qApp.quit)

		fileMenu.addAction(self.openAction)
		fileMenu.addAction(self.loadAction)
		fileMenu.addAction(self.saveAction)
		fileMenu.addSeparator()
		fileMenu.addAction(exitAction)

		# HEEEEEELP
		self.docAction = QtGui.QAction('Documentation..', self)
		self.docAction.triggered.connect(self.open_docs)
		self.bugAction = QtGui.QAction('Issues and Bugs..', self)
		self.bugAction.triggered.connect(self.open_issues)
		self.aboutAction = QtGui.QAction('About', self)

		helpMenu.addAction(self.docAction)
		helpMenu.addAction(self.bugAction)
		helpMenu.addAction(self.aboutAction)

		self.DM = DataModel.instance()
		self.DM.main_window = self
		self.LM = LayerManager.instance()
		self.LBLM = LabelManager.instance()

	def __del__(self):
		if self.DM is not None and self.DM.hdf5data is not None:
			log.info('+ Closing hdf5 file')
			self.DM.hdf5data.close()

	def open_docs(self):
		docURL = 'https://github.com/DiamondLightSource/SuRVoS/wiki'
		QtGui.QDesktopServices.openUrl(QtCore.QUrl(docURL))

	def open_issues(self):
		bugURL = 'https://github.com/DiamondLightSource/SuRVoS/issues'
		QtGui.QDesktopServices.openUrl(QtCore.QUrl(bugURL))

	def load_data(self):
		if self.main_widget.load_data_action():
			self.main_container.setCurrentIndex(1)

	def load_workspace(self, path=None):
		if type(path) not in [str, unicode]:
			msg = "Select the workspace folder"
			flags = QtGui.QFileDialog.ShowDirsOnly
			path = QtGui.QFileDialog.getExistingDirectory(self, msg, '.', flags)
			if path is None or len(path) == 0:
				return

		if self.main_widget.load_workspace(path):
			self.main_container.setCurrentIndex(1)

	def on_error(self, msg):
		QtGui.QMessageBox.critical(self, "Error", msg)
		self.overlay.post()

	def addWidget(self, widget, ptype=None):
		self.centralWidget().addWidget(widget, ptype)

	def resizeEvent(self, event):
		self.overlay.resize(event.size())
		event.accept()

	def closeEvent(self, event):
		log.info("\n### Destroying ###")
		event.accept()

	def keyPressEvent(self, event):
		modifiers = QtGui.QApplication.keyboardModifiers()
		if event.key() == QtCore.Qt.Key_Escape:
			self.setFocus()
		elif event.key() >= QtCore.Qt.Key_1 and \
				event.key() < QtCore.Qt.Key_1 + self.main_widget.leftPanel.count():
			index = event.key() - QtCore.Qt.Key_1
			if self.main_widget.leftPanel.isTabEnabled(index):
				self.main_widget.leftPanel.setCurrentIndex(index)
		elif modifiers == QtCore.Qt.ControlModifier and \
				event.key() == QtCore.Qt.Key_Z:
			# CTRL + Z
			log.debug('* DEBUG: CTRL+Z clicked')
			if self.DM.last_changes is not None:
				ds, slices, indexes, values, active_roi = self.DM.last_changes
				slice_z, slice_y, slice_x = slices

				hdata = self.DM.load_slices(ds, slice_z, slice_y, slice_x,
											apply_roi=active_roi)
				prev_values = hdata[indexes[:, 0], indexes[:, 1], indexes[:, 2]]
				hdata[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = values
				self.DM.write_slices(ds, hdata, slice_z, slice_y, slice_x,
									 apply_roi=active_roi)
				self.DM.last_changes = (ds, (slice_z, slice_y, slice_x),
										indexes, prev_values, active_roi)
				self.LM.update()
				log.info('+ Done')

	def mousePressEvent(self, event):
		self.setFocus()

class MainWidget(QtGui.QWidget):

	completed = QtCore.pyqtSignal()

	def __init__(self, title='QMainWindow'):
		super(MainWidget, self).__init__()
		self.setWindowTitle(title)

		self.pluginCount = 0

		self.setLayout(QtGui.QHBoxLayout())
		self.splitter = QtGui.QSplitter()
		self.layout().setContentsMargins(20,20,20,20)
		self.layout().addWidget(self.splitter)

		dirp = os.path.dirname(os.path.realpath(__file__))
		self.leftPanel = QtGui.QTabWidget(self)
		self.leftPanel.setTabPosition(QtGui.QTabWidget.West)
		self.leftPanel.setMaximumWidth(400)

		self.centerPanel = QtGui.QTabWidget()
		self.centerPanel.setTabPosition(QtGui.QTabWidget.West)

		self.splitter.addWidget(self.leftPanel)
		self.splitter.addWidget(self.centerPanel)

		self.launcher = Launcher.instance()
		self.DM = DataModel.instance()
		self.LM = LayerManager.instance()
		self.LBLM = LabelManager.instance()

		self.slice_viewer = None

	def __add__(self, plugin):
		return self.add_widget(plugin)

	def load_data_action(self):
		path = QtGui.QFileDialog.getOpenFileName(self, "Select input source",
												 filter='*.rec *.npy *.h5')
		if path is not None and len(path) > 0:
			dataset = None
			if path.endswith('.h5'):
				available_hdf5 = self.DM.available_hdf5_datasets(path)
				selected, accepted = ComboDialog.getOption(available_hdf5, parent=self)
				if accepted == QtGui.QDialog.Rejected:
					return
				dataset = selected

			self.load_data(str(path), dataset=dataset)
			return True
		return False

	def get_valid_workspace_folder(self):
		msg = "Select a folder to save the workspace"
		err_msg = "Workspace folder is not empty. Previously saved workspace" \
				  " files will be overwritten. Do you wish to continue?"
		flags = QtGui.QFileDialog.ShowDirsOnly
		wdir = QtGui.QFileDialog.getExistingDirectory(self, msg, '.', flags)

		do_continue = False
		while wdir != '' and len(os.listdir(wdir)) > 0 and not do_continue:
			ans = QtGui.QMessageBox.question(self, "Folder not empty", err_msg,
											 QtGui.QMessageBox.Yes,
											 QtGui.QMessageBox.No)
			if ans == QtGui.QMessageBox.No:
				wdir = QtGui.QFileDialog.getExistingDirectory(self, msg,
															  '.', flags)
			else:
				do_continue = True
		return wdir

	def load_data(self, path, dataset='data'):
		wdir = self.get_valid_workspace_folder()
		if wdir is None or len(wdir) == 0:
			return
		log.info("Workspace directory: {}".format(wdir))

		self.on_workspace(wdir)
		self.launcher.run(ac.load_data, path=path, dataset=dataset,
						  cb=self.on_data_loaded, caption='Loading data..')

	def on_data_loaded(self, result):
		self.DM.data = '/data'
		self.DM.data_shape = self.DM.ds_shape('/data')
		self.DM.active_roi = (slice(0, self.DM.data_shape[0]),
							  slice(0, self.DM.data_shape[1]),
							  slice(0, self.DM.data_shape[2]))
		attrs = self.DM.attrs('/data')

		# Left column
		self.visualization = Visualization()
		self.roi = ROI()
		self.SuperRegions = SuperRegions()
		self.voxel_channels = FeatureChannels()
		self.annotations = Annotations()
		self.training = Training()
		self.export = Export()

		self.addWidget(self.visualization, pvisible=True)

		self.addWidget(self.roi, pvisible=True, enabled=True, after=self)
		self.addWidget(self.voxel_channels, pvisible=False, \
					   enabled=True, after=self)
		self.addWidget(self.SuperRegions, pvisible=False, \
					   enabled=True, after=self)
		self.addWidget(self.training, pvisible=False, \
					   enabled=True, after=self)
		self.addWidget(self.annotations, pvisible=False, \
					   enabled=True, after=self)
		self.addWidget(self.export, pvisible=False, \
					   enabled=True, after=self)


		# Midle
		self.slice_viewer = SliceViewer()
		self.addWidget(self.slice_viewer)
		self.label_splitter = LabelSplitter()
		self.addWidget(self.label_splitter)
		self.level_statistics = LevelStats()
		self.addWidget(self.level_statistics)

		self.LM.clear()
		self.LM.addLayer(self.DM.data, 'Data', level='Data', cmap=attrs['cmap'],
						 vmin=attrs['evmin'], vmax=attrs['evmax'],
						 orient=self.DM.Axial)
		self.LM.update()

	def on_workspace(self, wdir):
		self.DM.wspath = wdir

		self.DM.main_window.openAction.setEnabled(False)
		self.DM.main_window.loadAction.setEnabled(False)
		self.DM.main_window.saveAction.setEnabled(True)

	def load_workspace(self, ws_path):
		self.DM.wspath = ws_path
		if self.on_load_workspace():
			self.on_workspace(ws_path)
			return True
		else:
			errmsg = 'No valid workspace found'
			QtGui.QMessageBox.critical(self, 'Error', errmsg)
		return False

	def save_backup(self):
		self.launcher.setup('+ Saving backups for levels: {}'
							.format(self.LBLM.levels()))
		for level in self.LBLM.levels():
			dataset = self.LBLM.dataset(level)
			log.info('+ Backing up [Level {}]: {}'.format(level, dataset))

			out_dataset = '{}.bak'.format(self.DM.ds_path(dataset))

			if os.path.isfile(out_dataset):
				err_msg = 'Backup for [Level {}] already exists, ' \
						  'do you want to overwrite it?'.format(level)
				ans = QtGui.QMessageBox.question(self, "Error", err_msg,
												 QtGui.QMessageBox.Yes,
												 QtGui.QMessageBox.No)
				if ans == QtGui.QMessageBox.No:
					log.info('  * Skipping.')
					continue

			data = self.DM.load_ds(dataset)
			attrs = self.DM.attrs(dataset)

			with h5.File(out_dataset, 'w') as f:
				out_ds = f.create_dataset('data', data=data)
				for k, v in attrs.items():
					out_ds.attrs[k] = v

			log.info('  * Done.')
		self.launcher.cleanup()

	def on_load_workspace(self):
		pdata = []

		if self.DM.has_ds('/data'):
			attrs = self.DM.attrs('/data')
			self.on_data_loaded(True)
			self.roi.loadROIs()
		else:
			return False

		if self.DM.has_grp('channels'):
			log.info('+ Loading Feature Channels')
			for fidx, desc in sorted(self.DM.available_channels(filter_active=False)):
				attrs = self.DM.attrs(desc)
				keys = attrs.keys()
				active = attrs['active']
				fname = attrs['feature_name']
				ftype = attrs['feature_type']
				keys.remove('feature_idx')
				keys.remove('feature_name')
				keys.remove('feature_type')
				keys.remove('active')
				params = {k: attrs[k] for k in keys}
				name = desc.split('/')[1]
				log.info('* Channel {} {}'.format(fname, ftype))
				self.voxel_channels.load_channel(fidx, name, ftype, active, params)

		if self.DM.has_ds('supervoxels/supervoxels') and \
				self.DM.has_ds('supervoxels/supervoxels_idx') and \
				self.DM.has_ds('supervoxels/supervoxels_table') and \
				self.DM.has_ds('supervoxels/graph_edges') and \
				self.DM.has_ds('supervoxels/graph_edge_weights'):
			log.info('+ Loading SuperVoxels')
			svds = 'supervoxels/supervoxels'
			svidx = 'supervoxels/supervoxels_idx'
			svtable = 'supervoxels/supervoxels_table'
			attrs = self.DM.attrs(svds)
			source = attrs['source']
			sp_shape = attrs['sp_shape']
			spacing = attrs['spacing']
			compactness = attrs['compactness']
			sptotal = attrs['num_supervoxels']
			self.SuperRegions.load_supervoxels(svds, svidx, svtable,
												source, sp_shape, spacing,
												compactness, sptotal)

		if self.DM.has_ds('megavoxels/megavoxels') and \
				self.DM.has_ds('megavoxels/megavoxels_idx') and \
				self.DM.has_ds('megavoxels/megavoxels_table'):
			log.info('+ Loading MegaVoxels')
			mvds = 'megavoxels/megavoxels'
			mvidx = 'megavoxels/megavoxels_idx'
			mvtable = 'megavoxels/megavoxels_table'
			attrs =  self.DM.attrs(mvds)
			source = attrs['source']
			lamda = attrs['lamda']
			nbins = attrs['nbins']
			gamma = attrs['gamma']
			mptotal = attrs['num_megavoxels']
			self.SuperRegions.load_megavoxels(mvds, mvidx, mvtable,
											   source, lamda, nbins, gamma,
											   mptotal)

		if self.DM.has_grp('annotations'):
			for levelid, dataset, active in sorted(self.DM.available_annotations()):
				if active != True:
					log.info('+ Found [Level {}] -- skipping'.format(levelid))
					self.LBLM.foundLevel(levelid)
					continue
				self.LBLM.loadLevel(levelid, dataset)

		log.info('+ Ready.')
		self.LM.update()
		return True


	def addWidget(self, plugin, ptype=None, pvisible=True, enabled=True, after=None):
		ptype = ptype or plugin.ptype

		if ptype == Plugin.Widget:
			self.centerPanel.addTab(plugin, plugin.name)
			self.centerPanel.currentChanged.connect(plugin.on_tab_changed)
		elif ptype == Plugin.Plugin:
			plugin.tab_idx = self.leftPanel.count()
			self.leftPanel.currentChanged.connect(plugin.on_tab_changed)
			self.leftPanel.addTab(plugin, plugin.name)
			self.pluginCount += 1
		return self


class Overlay(QtGui.QWidget):

	def __init__(self, title='Loading...', parent=None):
		QtGui.QWidget.__init__(self, parent)
		palette = QtGui.QPalette(self.palette())
		palette.setColor(palette.Background, QtCore.Qt.transparent)

		self.setPalette(palette)
		self.title = title
		self.hide()
		self.timer = None

	def pre(self, caption):
		self.title = caption
		self.show()
		time.sleep(0.5)

	def paintEvent(self, event):
		painter = QtGui.QPainter()
		painter.begin(self)
		painter.setRenderHint(QtGui.QPainter.Antialiasing)
		painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
		painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))

		for i in range(6):
			if self.counter == i:
				painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 50)))
			else:
				painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
			painter.drawEllipse(
				self.width()/2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
				self.height()/2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
				20, 20)

			font = painter.font()
			font.setPointSize(16)
			painter.setFont(font)
			painter.setPen(QtGui.QColor(80, 80, 80));
			painter.drawText(self.width()//2-200, self.height()//2-40, 400, 200,\
							 QtCore.Qt.AlignCenter, self.title)

		painter.end()

	def showEvent(self, event):
		self.counter = 0
		self.timer = self.startTimer(80)

	def timerEvent(self, event):
		self.counter += 1
		self.counter %= 6
		self.update()

	def post(self):
		log.info('+ Ready')
		self.hide()

	def resizeEvent(self, ev):
		#self.stop.move(self.width()//2-50, self.height()//2 + 100)
		pass

	def cancel_task(self):
		Launcher.instance().terminate()
