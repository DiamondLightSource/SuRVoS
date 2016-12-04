

from ..qt_compat import QtGui, QtCore

from .base import HWidgets, HeaderLabel, FileWidget, CheckableCombo, \
                  ComboDialog

from .mpl_widgets import MplCanvas

from ..core import DataModel, Launcher

from ..actions import volread

import numpy as np
import os

class LoadWidget(QtGui.QWidget):

    data_loaded = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(LoadWidget, self).__init__(parent)

        self.DM = DataModel.instance()
        self.launcher = Launcher.instance()
        self.data = None

        main_layout = QtGui.QVBoxLayout()
        main_layout.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setLayout(main_layout)

        container = QtGui.QWidget(self)
        hbox = QtGui.QHBoxLayout(container)
        container.setMaximumWidth(950)
        container.setMaximumHeight(530)
        container.setLayout(hbox)
        container.setObjectName("loaderContainer")
        container.setStyleSheet('QWidget#loaderContainer {'
                                '  background-color: #fefefe; '
                                '  border-radius: 10px;'
                                '}')
        lvbox = QtGui.QVBoxLayout()
        rvbox = QtGui.QVBoxLayout()
        lvbox.setAlignment(QtCore.Qt.AlignTop)
        rvbox.setAlignment(QtCore.Qt.AlignTop)
        hbox.addLayout(lvbox, 1)
        hbox.addLayout(rvbox, 1)

        main_layout.addWidget(container)

        ################################################
        # LEFT
        ################################################
        lvbox.addWidget(HeaderLabel('Preview Dataset'))

        self.wperspective = QtGui.QComboBox()
        self.wperspective.addItem('Axial')
        self.wperspective.addItem('Coronal')
        self.wperspective.addItem('Sagittal')

        self.winvert = CheckableCombo()
        self.winvert.addItem('Invert Z')
        self.winvert.addItem('Invert Y')
        self.winvert.addItem('Invert X')
        self.winvert.addItem('Transpose X <-> Y')

        self.slider = QtGui.QSlider(1)
        lvbox.addWidget(self.slider)

        lvbox.addWidget(HWidgets('3D Perspective:', self.wperspective,
                                 'Axis Order:', self.winvert,
                                 stretch=[0,1,0,1]))

        self.canvas = MplCanvas()
        lvbox.addWidget(self.canvas)

        ################################################
        # RIGHT
        ################################################

        # INPUT
        rvbox.addWidget(HeaderLabel('Input Dataset'))

        self.winput = FileWidget(extensions='*.mrc *.rec *.npy *.h5 *.hdf5 *.tif *.tiff', save=False)
        rvbox.addWidget(HWidgets('Select File:', self.winput, stretch=[0,1]))

        # OUTPUT
        rvbox.addWidget(HeaderLabel('Workspace Folder'))
        self.woutput = FileWidget(folder=True)

        rvbox.addWidget(HWidgets('Select Folder:', self.woutput, stretch=[0,1]))
        rvbox.addWidget(QtGui.QWidget(), 1)

        # Save | Cancel
        self.cancel = QtGui.QPushButton('Cancel')
        self.cancel.setMinimumWidth(100)
        self.cancel.setMinimumHeight(70)
        self.load = QtGui.QPushButton('Load')
        self.load.setMinimumWidth(100)
        self.load.setMinimumHeight(70)
        rvbox.addWidget(HWidgets(None, self.cancel, self.load, stretch=[1,0,0]))

        ###########################

        self.winput.path_updated.connect(self.load_data)
        self.wperspective.currentIndexChanged.connect(self.on_perspective)
        self.winvert.selectionChanged.connect(self.on_axis)
        self.slider.valueChanged.connect(self.update_image)
        self.load.clicked.connect(self.on_load_data)

    def load_data(self, path):
        if path is not None and len(path) > 0:
            dataset = None
            if path.endswith('.h5') or path.endswith('.hdf5'):
                available_hdf5 = self.DM.available_hdf5_datasets(path)
                selected, accepted = ComboDialog.getOption(available_hdf5, parent=self)
                if accepted == QtGui.QDialog.Rejected:
                    return
                dataset = selected

            self.launcher.run(volread, path=path, dataset=dataset,
                              caption='Loading data...',
                              cb=self.on_data_loaded)

    def on_data_loaded(self, result):
        data, vmin, vmax, evmin, evmax = result
        if data is None or data.ndim != 3:
            raise Exception('SuRVoS can only process 3 dimensional data: {}'.format(data.shape))

        self.DM.tmp_data = data
        self.DM.tmp_stats = (vmin, vmax, evmin, evmax)
        self.update_image(idx=None)

    def transform_data(self):
        tidx = self.wperspective.currentIndex()
        trans = [0,1,2]
        tchanged = False
        if tidx != 0:
            trans = [1,0,2] if tidx == 1 else [2,0,1]
            tchanged = True

        inv_indexes = self.winvert.getSelectedIndexes()
        if 3 in inv_indexes:
            trans[1], trans[2] = trans[2], trans[1]
            tchanged = True

        if tchanged:
            data = np.transpose(self.DM.tmp_data, trans)
        else:
            data = self.DM.tmp_data

        if 0 in inv_indexes:
            data = data[::-1]
        if 1 in inv_indexes:
            data = data[:, ::-1]
        if 2 in inv_indexes:
            data = data[:, :, ::-1]

        return data

    def on_axis(self, *args):
        self.update_image()

    def on_perspective(self, *args):
        self.update_image()

    def update_image(self, idx=None):
        if idx is None:
            self.data = self.transform_data()
            idx = self.data.shape[0]//2
            self.slider.blockSignals(True)
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.data.shape[0])
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self.canvas.ax.set_ylim([self.data.shape[1] + 1, -1])
            self.canvas.ax.set_xlim([-1, self.data.shape[2] + 1])

        evmin, evmax = self.DM.tmp_stats[-2:]

        img = self.data[idx]
        self.canvas.ax.imshow(img, 'gray', vmin=evmin, vmax=evmax)
        self.canvas.ax.grid(False)
        self.canvas.redraw()

    def on_load_data(self):
        wspath = self.woutput.value()
        if self.data is not None and wspath is not None:
            if not os.path.isdir(wspath):
                errmsg = 'Workspace Path not valid: {}'.format(wspath)
                QtGui.QMessageBox.critical(self, 'Error', errmsg)
                return

            do_continue = True
            if len(os.listdir(wspath)) > 0:
                errmsg = 'Workspace folder is not empty, do you still want to continue? It might overwrite and remove previous files.'
                ans = QtGui.QMessageBox.question(self, "Workspace Folder not empty",
                                                 errmsg,
                                                 QtGui.QMessageBox.Yes,
                                                 QtGui.QMessageBox.No)
                if ans == QtGui.QMessageBox.No:
                    do_continue = False

            if do_continue:
                self.DM.tmp_data = np.ascontiguousarray(self.data)
                del self.data
                self.data_loaded.emit(wspath)
        else:
            if self.data is None:
                errmsg = 'Data has to be loaded first'
                QtGui.QMessageBox.critical(self, 'Error', errmsg)
                return
            if wspath is None:
                errmsg = 'Workspace path has to be set first'
                QtGui.QMessageBox.critical(self, 'Error', errmsg)
                return
