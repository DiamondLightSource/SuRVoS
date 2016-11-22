
from ..qt_compat import QtGui, QtCore

import matplotlib as mpl
mpl.use("Qt4Agg")

import matplotlib.pyplot as plt

import numpy as np

from .base import WSliderPane, SComboBox, ULabel

from ..core import DataModel, LayerManager, LabelManager, Launcher
from ..plugins.base import Plugin

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4 import FigureManagerQT
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg


Axial = DataModel.instance().Axial
Sagittal = DataModel.instance().Sagittal
Coronal = DataModel.instance().Coronal

views = [Axial, Sagittal, Coronal]


class OrthogonalViewer(Plugin):

    name = 'Orthogonal Viewer'

    def __init__(self, ptype=Plugin.Widget):
        super(OrthogonalViewer, self).__init__(ptype=ptype)

        for i, orient in enumerate(views):
            mc = MplCanvas(orient=orient)
            self.layout.addWidget(mc, i%2, i//2)

class MplCanvas(QtGui.QWidget):

    def __init__(self, orient=Axial, axisoff=True, autoscale=False, **kwargs):
        super(MplCanvas, self).__init__()

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.launcher = Launcher.instance()

        self.orient = orient
        self.idx = self.DM.data_shape[orient]//2
        self.setLayout(QtGui.QVBoxLayout())

        # Figure
        self.fig, self.ax, self.canvas = self.figimage(axisoff=axisoff)
        self.pressed = False
        self.ax.autoscale(enable=autoscale)
        self.layout().addWidget(self.canvas, 1)
        self.canvas.mpl_connect('button_press_event',self.on_press)

    def replot(self):
        self.ax.clear()
        self.ax.cla()
        self.ax.spines['left'].set_position(('outward', 3))
        self.ax.spines['bottom'].set_position(('outward', 3))
        self.ax.spines['right'].set_position(('outward', 3))
        self.ax.spines['top'].set_position(('outward', 3))

    def update_slides(self):
        self.slider.setMaximum(self.DM.data_shape[self.orient]-1)
        self.slider.setValue(self.DM.data_shape[self.orient]//2)

    def redraw(self):
        self.canvas.draw_idle()

    def figimage(self, scale=1, dpi=None, axisoff=True):
        fig = plt.figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        if axisoff:
            fig.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.99)
        canvas.draw()
        return fig, ax, canvas

    def on_press(self, event):
        self.setFocus()

class PerspectiveCanvas(MplCanvas):

    def __init__(self, orient=Axial, axisoff=True, autoscale=False, **kwargs):
        super(PerspectiveCanvas, self).__init__(axisoff=axisoff,
                                                autoscale=autoscale)

        topbox = QtGui.QHBoxLayout()
        self.layout().insertLayout(0, topbox)

        # Slider
        self.slider = WSliderPane(self.DM.data_shape[self.orient]-1, orient=1)
        self.slider.setTracking(True)
        topbox.addWidget(self.slider, 1)

        # Slider Text
        self.text_idx = ULabel(str(self.idx))
        topbox.addWidget(self.text_idx)

        # Slots
        self.DM.cropped.connect(self.update_slides)

    def on_perspective_changed(self, orient):
        self.orient = orient

        for layer in self.LM.layers():
            layer.orient = orient

        self.blockSignals(True)
        self.slider.setMaximum(self.DM.data_shape[self.orient]-1)
        self.idx = (self.DM.data_shape[self.orient]) // 2
        self.slider.setValue(self.idx)
        self.blockSignals(False)
        self.text_idx.setText(str(self.idx))

        self.replot()

    def on_source_changed(self, idx):
        pass

class ConfidenceCanvas(MplCanvas):

    def __init__(self, k=None, name=None, *args, **kwargs):
        super(ConfidenceCanvas, self).__init__(*args, **kwargs)
        self.name = name
        self.data = self.DM.splabels
        self.k = k
        self.replot()

    def replot(self):
        self.ax.clear()
        slice = self.get_slice(self.idx)
        self.ax.imshow(slice, 'optiona', vmin=0, vmax=1)
        self.ax.set_title('P(%s)' % self.name)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.redraw()

    def get_slice(self, idx):
        if self.k is None:
            shape = list(self.DM.data_shape)
            del shape[self.orient]
            data = np.zeros(shape, int)
            return data if self.orient < 2 else data.T

        if self.orient == Axial:
            data = self.data[idx, :, :]
        elif self.orient == Sagittal:
            data = self.data[:, idx, :]
        else:
            data = self.data[:, :, idx].T

        return self.DM.spprobabilities[:, self.k][data]

    def update_volume(self, idx=None):
        if idx is not None:
            self.idx = idx
        slide = self.get_slice(self.idx)
        self.ax.images[0].set_array(slide)
        self.redraw()


class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, figure, **kwargs):
        self.fig = figure
        super(FigureCanvas, self).__init__(self.fig)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                            QtGui.QSizePolicy.Expanding)
        self.updateGeometry()
        self.mouse_lines = None
        self.mouse_color = QtCore.Qt.black
        self.mouse_width = 1

    def resizeEvent(self, event):
        super(FigureCanvas, self).resizeEvent(event)
        self.resize_event()

    def setColor(self, color):
        self.mouse_color = color

    def setMouseWidth(self, width):
        self.mouse_width = width

    def updateMouseLines(self, y, x):
        y = self.height() - int(y)
        x = int(x)
        if self.mouse_lines is None:
            self.mouse_lines = [[y], [x]]
        else:
            self.mouse_lines[0].append(y)
            self.mouse_lines[1].append(x)
        self.update()

    def setMouseLines(self, lines):
        self.mouse_lines = lines
        self.update()

    def paintEvent(self, event):
        super(FigureCanvas, self).paintEvent(event)

        if self.mouse_lines is None:
            return

        qp = QtGui.QPainter()
        qp.begin(self)

        qp.setPen(QtGui.QPen(self.mouse_color, self.mouse_width))

        for i in range(len(self.mouse_lines[0]) - 1):
            y0, x0 = self.mouse_lines[0][i], self.mouse_lines[1][i]
            y1, x1 = self.mouse_lines[0][i+1], self.mouse_lines[1][i+1]
            qp.drawLine(x0, y0, x1, y1)

        qp.end()
