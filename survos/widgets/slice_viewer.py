

from ..qt_compat import QtGui, QtCore, QtWidgets

import time

import h5py as h5
import numpy as np
import logging as log

import os
import matplotlib

from .. import actions as ac

from scipy.ndimage.interpolation import zoom
from scipy import ndimage as ndi
from skimage.morphology import disk

from ..plugins import Plugin
from ..plugins.visualization import LayersMenu, ContrastMenu
from ..core import DataModel, LayerManager, LabelManager, Launcher
from .mpl_widgets import PerspectiveCanvas

from .base import TComboBox, PLineEdit, HWidgets, HSize3D, ULabel, OddSlider, \
                  SourceCombo

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from skimage.draw import line, bezier_curve


class NavigationToolbar(NavigationToolbar2QT):

    home_pressed = QtCore.pyqtSignal()
    toggle_grid = QtCore.pyqtSignal()

    toolitems = (
        (None, None, None, None),
        ('Home', 'Reset original view', 'home', 'home'),
        (None, None, None, None),
        ('Draw', 'Draw ground truth seeds', 'pencil', 'gt'),
        #('Region', 'Grow a region', 'region2', 'region'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'enlarge', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom'),
        (None, None, None, None),
        ('Grid', 'Turn on/off the grid', 'grid', 'grid'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'floppy-disk', 'save_figure'),
    )

    icondir = os.path.dirname(os.path.realpath(__file__)) + '/../images/PNG/'

    def __init__(self, canvas, parent=None, locLabel=None):
        super(NavigationToolbar, self).__init__(canvas, parent=parent)
        self.parent = parent
        self.locLabel = locLabel
        self.setOrientation(QtCore.Qt.Vertical)

    def gt(self):
        if self._active == 'GT':
            self._active = None
        else:
            self._active = 'GT'
        self._update_buttons_checked()

        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

    def region(self):
        if self._active == 'REGION':
            self._active = None
        else:
            self._active = 'REGION'
        self._update_buttons_checked()

        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

    def _icon(self, name):
        icon = QtGui.QIcon(os.path.abspath(os.path.join(self.icondir, name+'.png')))
        return icon

    def _init_toolbar(self):

        con_button = QtWidgets.QToolButton()
        con_action = QtWidgets.QAction(self._icon('contrast'), '', con_button)
        con_button.setDefaultAction(con_action)
        con_menu = ContrastMenu(self)
        con_button.setMenu(con_menu)
        con_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        con_button.setStyleSheet('QToolButton {margin-left: 0px; width: 24px;}'
                                 'QToolButton::menu-indicator {'
                                 '  width: 0px; border: none; image: none;'
                                 '}')
        self.addWidget(con_button)

        viz_button = QtWidgets.QToolButton()
        viz_action = QtWidgets.QAction(self._icon('layers'), '', viz_button)
        viz_button.setDefaultAction(viz_action)
        viz_menu = LayersMenu(self)
        viz_button.setMenu(viz_menu)
        viz_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        viz_button.setStyleSheet('QToolButton {margin-left: 0px; width: 24px;}'
                                 'QToolButton::menu-indicator {'
                                 '  width: 0px; border: none; image: none;'
                                 '}')
        self.addWidget(viz_button)

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                icon = self._icon(image_file)
                a = self.addAction(icon, text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan', 'gt', 'region', 'grid']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)

        #self.buttons = {}

        #self.adj_window = None

    def _update_buttons_checked(self):
        self._actions['pan'].setChecked(self._active == 'PAN')
        self._actions['zoom'].setChecked(self._active == 'ZOOM')
        self._actions['gt'].setChecked(self._active == 'GT')
        #self._actions['region'].setChecked(self._active == 'REGION')
        self.parent.show_tools(self._active)

    def grid(self):
        self.toggle_grid.emit()

    def layers(self):
        self.vizmenu.exec_()

    def home(self):
        self.home_pressed.emit()


class SliceViewer(Plugin):
    name = 'Slice Viewer'

    def __init__(self, data=None, ptype=Plugin.Widget):
        super(SliceViewer, self).__init__(ptype=ptype)

        self.DM = DataModel.instance()
        self.layered_canvas = LayeredCanvas()

        self.tool_options = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout()
        self.tool_options.setLayout(hbox)
        hbox.setContentsMargins(0, 0, 0, 0)

        self.locLabel = QtWidgets.QLabel("", self)
        self.locLabel.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.locLabel.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                              QtWidgets.QSizePolicy.Ignored))

        self.toolbar = NavigationToolbar(
            self.layered_canvas.canvas, self, self.locLabel)
        self.toolbar.setMaximumWidth(50)
        self.toolbar.toggle_grid.connect(self.layered_canvas.toggle_grid)
        self.toolbar.home_pressed.connect(self.on_home_pressed)

        self.layered_canvas.toolbar = self.toolbar
        self.source_combo = SourceCombo(listen_DM=True, listen_self=True)

        self.layout.addWidget(HWidgets(self.source_combo,
                                       self.tool_options,
                                       self.locLabel,
                                       stretch=[0, 1, 0]))

        self.layout.addWidget(HWidgets(self.toolbar,
                                       self.layered_canvas,
                                       stretch=[0, 1]))

        self.gt_tool = GtTool()
        self.grow_tool = GrowTool()

    def on_home_pressed(self):
        sy, sx = self.DM.region_shape()[1:]
        self.layered_canvas.ax.set_ylim([sy, 0])
        self.layered_canvas.ax.set_xlim([0, sx])
        self.layered_canvas.redraw()

    def increase_slider(self):
        self.layered_canvas.increase_slider()

    def decrease_slider(self):
        self.layered_canvas.decrease_slider()

    def selectSource(self, source):
        self.layered_canvas.combo_source.setCurrentIndex(source)

    def show_tools(self, tool):
        layout = self.tool_options.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

        if tool == 'GT':
            layout.addWidget(self.gt_tool)
        elif tool == 'REGION':
            layout.addWidget(self.grow_tool)


class LayeredCanvas(PerspectiveCanvas):

    def __init__(self, *args, **kwargs):
        super(LayeredCanvas, self).__init__(*args, **kwargs)
        self.DM.evmin_changed.connect(self.update_vmin)
        self.DM.evmax_changed.connect(self.update_vmax)

        # self.LM.added.connect(self.replot)
        # self.LM.removed.connect(self.replot)
        self.LM.updated.connect(self.replot)
        self.LM.opacity.connect(self.on_layer_opacity)
        # self.LM.toggled.connect(self.on_layer_toggled)

        self.LBLM.labelUpdated.connect(self.replot)

        self.canvas.mpl_connect('button_press_event', self.canvas_onmousepress)
        self.canvas.mpl_connect('button_release_event',
                                self.canvas_onmouserelease)
        self.canvas.mpl_connect('motion_notify_event', self.canvas_onmousemove)
        self.canvas.mpl_connect('scroll_event', self.canvas_onscroll)

        self.slider.valueChanged.connect(self.update_volume)

        self.idx = self.DM.data_shape[0] // 2
        self.DM.current_idx = self.idx

        self.DM.roi_changed.connect(self.update_axes)


        self.layout().setContentsMargins(0, 0, 0, 0)

        self.current = None

        # Draw!
        self.grid = False
        self.replot()

    def __del__(self):
        self.ax.close()

    def update_axes(self):
        sz, sy, sx = self.DM.active_roi
        self.slider.setMinimum(0)
        self.slider.setMaximum(sz.stop - sz.start - 1)
        self.slider.setValue((sz.stop - sz.start) // 2)
        self.ax.set_ylim([sy.stop - sy.start + 1, -1])
        self.ax.set_xlim([-1, sx.stop - sx.start + 1])

    def get_idx(self):
        return self.idx

    def on_layer_opacity(self, name, level, val):
        idx = self.LM.index(str(name), str(level))
        if idx < 0:
            return
        self.ax.images[idx].set_alpha(val)
        self.redraw()

    def on_layer_toggled(self, name, level, bol):
        self.replot()

    def replot(self):
        super(LayeredCanvas, self).replot()
        ylim, xlim = self.ax.get_ylim(), self.ax.get_xlim()
        valid = ylim != (0., 1.) and xlim != (0., 1.)
        for i, layer in enumerate(self.LM.visible_layers()):
            layer.draw(self.ax, self.idx, i)

        if valid:
            h, w = self.DM.data_shape[1:]
            x = [-2, w + 1, w + 1, -2, -2]
            y = [-2, -2, h + 1, h + 1, -2]
            self.ax.plot(x, y, 'r-', linewidth=5)
            self.ax.set_ylim(ylim)
            self.ax.set_xlim(xlim)

        self.ax.grid(self.grid)
        self.redraw()

    def update_vmin(self, val):
        data = self.LM.get('Data', 'Data')
        data.vmin = val
        if not data.binarize:
            self.ax.images[0].set_clim(vmin=val)
            self.redraw()
        else:
            self.LM.update()

    def update_vmax(self, val):
        data = self.LM.get('Data', 'Data')
        data.vmax = val
        if not data.binarize:
            self.ax.images[0].set_clim(vmax=val)
            self.redraw()
        else:
            self.LM.update()

    def increase_slider(self):
        if self.slider.value() < self.slider.maximum():
            self.slider.setValue(self.slider.value() + 1)

    def decrease_slider(self):
        if self.slider.value() > self.slider.minimum():
            self.slider.setValue(self.slider.value() - 1)

    def update_volume(self, idx):
        self.DM.current_idx = idx
        self.idx = idx
        self.text_idx.setText(str(idx))
        for (layer, image) in zip(self.LM.visible_layers(), self.ax.images):
            layer.update(image, self.idx)
        self.redraw()

    def canvas_onscroll(self, ev):
        if ev.button == 'up':
            self.increase_slider()
        else:
            self.decrease_slider()

    def canvas_onmousepress(self, ev):
        if self.DM.gtselected is None or self.toolbar._active != 'GT':
            return
        if ev.inaxes != self.ax:
            return
        self.pressed = True

        y = int(ev.ydata)
        x = int(ev.xdata)
        self.current = [[y], [x]]

        # Extract correct radius for current scale
        brush_width = self.DM.gtradius * 2 + 1
        bbox = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        canvas_width = bbox.width * self.fig.dpi
        xlim = self.ax.get_xlim()
        zoom_width = xlim[1] - xlim[0] - 2
        brush_width *=  canvas_width / zoom_width

        self.canvas.setMouseWidth(brush_width)
        self.canvas.updateMouseLines(ev.y, ev.x)

    def canvas_onmousemove(self, ev):
        if not self.pressed or ev.inaxes != self.ax:
            return
        y = int(ev.ydata)
        x = int(ev.xdata)
        self.current[0] += [y]
        self.current[1] += [x]
        self.canvas.updateMouseLines(ev.y, ev.x)

    def canvas_onmouserelease(self, ev):
        if self.DM.gtselected is None:
            return

        if self.pressed and self.toolbar._active == 'GT':
            self.pressed = False
            self.update_gt(ev)
        elif self.toolbar._active == 'REGION' and ev.inaxes == self.ax:
            self.calculate_region(ev)

        self.current = None
        self.canvas.setMouseLines(None)

    def calculate_region(self, ev):
        y = int(ev.ydata)
        x = int(ev.xdata)
        pos = (self.idx, y, x)
        source, lamda, bbox, region = self.DM.growing_bbox
        if self.DM.attrs('data/{}'.format(source), 'active') != 1:
            QtWidgets.QMessageBox.critical(self, 'Error',
                                       '{} dataset not valid'.format(source))
            return

        if region == 0:
            ldata = self.DM.gtselected['data']
            label = self.DM.gtselected['label']
            ds = self.DM.load_ds('data/{}'.format(source))
            self.launcher.run(ac.grow_voxels, dataset=ds, pos=pos, bbox=bbox,
                              lamda=lamda, annotations=ldata, label=label,
                              caption='Growing region..', cb=self.on_region_grow)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Not supported yet.")

    def on_region_grow(self, params):
        indexes, values = params
        self.DM.gtselected['modified'] = True
        level = self.DM.gtselected['level']
        label = self.DM.gtselected['label']
        data = self.DM.gtselected['data']
        self.DM.last_changes.append((level, label, indexes, values))
        self.LM.update()

    def update_gt(self, ev):
        data_shape = self.DM.data_shape
        level = self.DM.gtselected['level']
        label = self.DM.gtselected['label']
        plevel = self.DM.gtselected['parent_level']
        plabel = self.DM.gtselected['parent_label']

        t0 = time.time()

        # Get positions
        log.debug('* Smoothing indexes')
        cr, cc = self.current
        pr, pc = [], []
        if self.DM.gtinterpolation == 'bezier':
            for i in range(1, len(cr) - 1, 2):
                line_r, line_c = bezier_curve(cr[i - 1], cc[i - 1],
                                              cr[i], cc[i],
                                              cr[i + 1], cc[i + 1],
                                              2)
                pr.extend(line_r)
                pc.extend(line_c)
            if len(cr) % 2 == 0:
                line_r, line_c = line(cr[-2], cc[-2], cr[-1], cc[-1])
                pr.extend(line_r)
                pc.extend(line_c)
        else:
            for i in range(len(cr) - 1):
                line_r, line_c = line(cr[i], cc[i], cr[i + 1], cc[i + 1])
                pr.extend(line_r)
                pc.extend(line_c)

        pos = np.c_[pr, pc]

        if pos.shape[0] == 0:
            pos = np.c_[cr, cc]

        print("Time Smooth:", time.time() - t0)
        t0 = time.time()

        _, region_maxy, region_maxx = self.DM.region_shape()
        pos[:, 0] = np.clip(pos[:, 0], 0, region_maxy - 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, region_maxx - 1)

        radius = self.DM.gtradius

        ymin, xmin = pos.min(0) - radius
        ymax, xmax = pos.max(0) + radius

        ymin = max(ymin, 0)
        xmin = max(xmin, 0)
        ymax = min(ymax, region_maxy - 1)
        xmax = min(xmax, region_maxx - 1)

        mask2d = np.zeros((ymax-ymin+1, xmax-xmin+1), np.bool)
        mask2d[pos[:, 0] - ymin, pos[:, 1] - xmin] = True
        mask2d = ndi.binary_dilation(mask2d, disk(radius).astype(np.bool))

        slice_z = slice(self.idx, self.idx+1)
        slice_y = slice(ymin, ymax+1)
        slice_x = slice(xmin, xmax+1)

        pos = np.column_stack(np.where(mask2d))

        print("Time Dilate:", time.time() - t0)

        t0 = time.time()

        log.debug('* Calculating indexes')
        if self.DM.gtlevel == 1:

            t0 = time.time()

            sv = self.DM.load_slices(self.DM.svlabels, slice_z, slice_y, slice_x,
                                     apply_roi=True)
            sv = sv[0, pos[:, 0], pos[:, 1]]
            sv = np.unique(sv)
            if np.isnan(sv).any() or (sv < 0).any():
                self.launcher.show_error('SuperVoxels are not created for this ROI')
                return

            total_idx = np.array([], np.int32)

            with h5.File(self.DM.ds_path(self.DM.svtable), 'r') as t1, \
                 h5.File(self.DM.ds_path(self.DM.svindex), 'r') as t2:

                svtable = t1['data']
                svindex = t2['data']

                for v in sv:
                    if v == self.DM.svtotal - 1:
                        slc = slice(svtable[v], None)
                    else:
                        slc = slice(svtable[v], svtable[v+1])
                    total_idx = np.append(total_idx, svindex[slc])

            indexes = np.column_stack(np.unravel_index(total_idx, self.DM.data_shape))
            if self.DM.data_shape != self.DM.region_shape():
                log.debug('* Filtering indexes')
                mask = np.ones(indexes.shape[0], np.bool)
                for i, axis in enumerate(self.DM.active_roi):
                    curr = indexes[:, i]
                    mask[curr < axis.start] = False
                    mask[curr >= axis.stop] = False
                indexes = indexes[mask]

            zmin, ymin, xmin = indexes.min(0)
            zmax, ymax, xmax = indexes.max(0)
            indexes[:, 0] -= zmin
            indexes[:, 1] -= ymin
            indexes[:, 2] -= xmin
            apply_roi = False
        elif self.DM.gtlevel == 2:
            mv = self.DM.load_slices(self.DM.mvlabels, slice_z, slice_y, slice_x)
            mv = mv[0, pos[:, 0], pos[:, 1]]
            mv = np.unique(mv)
            if np.isnan(mv).any() or (mv < 0).any():
                self.launcher.show_error('MegaVoxels are not created for this ROI')
                return

            total_idx = np.array([], np.int32)

            with h5.File(self.DM.ds_path(self.DM.mvtable), 'r') as t1, \
                 h5.File(self.DM.ds_path(self.DM.mvindex), 'r') as t2:

                mvtable = t1['data']
                mvindex = t2['data']

                for v in mv:
                    if v == self.DM.mvtotal - 1:
                        slc = slice(mvtable[v], None)
                    else:
                        slc = slice(mvtable[v], mvtable[v+1])
                    total_idx = np.append(total_idx, mvindex[slc])

            indexes = np.column_stack(np.unravel_index(total_idx, self.DM.data_shape))
            if self.DM.data_shape != self.DM.region_shape():
                log.debug('* Filtering indexes')
                mask = np.ones(indexes.shape[0], np.bool)
                for i, axis in enumerate(self.DM.active_roi):
                    curr = indexes[:, i]
                    mask[curr < axis.start] = False
                    mask[curr >= axis.stop] = False
                indexes = indexes[mask]

            zmin, ymin, xmin = indexes.min(0)
            zmax, ymax, xmax = indexes.max(0)
            indexes[:, 0] -= zmin
            indexes[:, 1] -= ymin
            indexes[:, 2] -= xmin
            apply_roi = False
        else:
            zmin = zmax = self.idx
            indexes = np.empty((pos.shape[0], 3), np.int32)
            indexes[:, 0] = 0
            indexes[:, 1:] = pos
            apply_roi = True

        print("Time Get Index:", time.time() - t0)

        t0 = time.time()

        slice_z = slice(zmin, zmax+1)
        slice_y = slice(ymin, ymax+1)
        slice_x = slice(xmin, xmax+1)

        if plevel is not None and plevel >= 0 and plabel >= 0:
            plevel = self.LBLM.dataset(plevel)
            pdata = self.DM.load_slices(plevel, slice_z, slice_y, slice_x, apply_roi=apply_roi)
            mask = (pdata == plabel)
            valid = mask[indexes[:, 0], indexes[:, 1], indexes[:, 2]]
            indexes = indexes[valid]

        hdata = self.DM.load_slices(level, slice_z, slice_y, slice_x, apply_roi=apply_roi)
        values = hdata[indexes[:, 0], indexes[:, 1], indexes[:, 2]]
        hdata[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = label
        self.DM.write_slices(level, hdata, slice_z, slice_y, slice_x, apply_roi=apply_roi)
        self.DM.last_changes.append((level, (slice_z, slice_y, slice_x),
                                indexes, values, apply_roi))

        print("Time Replace Data:", time.time() - t0)

        log.debug('* Done')
        self.update_volume(self.idx)

    def toggle_grid(self):
        self.grid = not self.grid
        self.ax.grid(self.grid)
        self.redraw()


class GtTool(HWidgets):

    def __init__(self):
        items = ['Voxels', 'SuperVoxels']
        self.DM = DataModel.instance()
        self.launcher = Launcher.instance()
        self.alevel_combo = TComboBox('Annotation Level:', items,
                                      selected=self.DM.gtlevel)
        self.radius_slider = OddSlider('', vmin=1, vmax=31, current=1,
                                       interval=2, lblwidth=30)
        self.interp_type = TComboBox('Interpolation:', ['linear', 'bezier'])
        super(GtTool, self).__init__(self.alevel_combo, 'Width:',
                                     self.radius_slider, self.interp_type, None,
                                     stretch=[0, 0, 0, 0, 1])

        self.alevel_combo.currentIndexChanged.connect(self.on_annotation_level)
        self.radius_slider.valueChanged.connect(self.on_radius)
        self.interp_type.currentIndexChanged.connect(self.on_interpolation)

    def on_annotation_level(self, idx):
        if idx == 1 and self.DM.svlabels is None:
            self.launcher.error.emit('SuperVoxels need to be created first')
            self.alevel_combo.setCurrentIndex(self.DM.gtlevel)
        else:
            self.DM.gtlevel = idx

    def on_interpolation(self):
        self.DM.gtinterpolation = self.interp_type.currentText()

    def on_radius(self, idx):
        self.DM.gtradius = int(self.radius_slider.value()) // 2


class GrowTool(HWidgets):

    def __init__(self):
        self.DM = DataModel.instance()
        self.launcher = Launcher.instance()

        z, y, x = self.DM.growing_bbox[2]
        self.bboxz = PLineEdit(z, parse=int)
        self.bboxz.setMaximumWidth(50)
        self.bboxy = PLineEdit(y, parse=int)
        self.bboxy.setMaximumWidth(50)
        self.bboxx = PLineEdit(x, parse=int)
        self.bboxx.setMaximumWidth(50)

        self.bbox_lamda = PLineEdit(self.DM.growing_bbox[1], parse=float)
        self.bbox_lamda.setMaximumWidth(50)
        items = ['data', 'stretch', 'smooth', 'denoised']
        self.bbox_source = TComboBox('Source:', items,
                                     selected=items.index(self.DM.growing_bbox[0]))
        items = ['Voxels', 'SuperVoxels', 'MegaVoxels']
        self.alevel_combo = TComboBox('Annotation Level:', items,
                                      selected=self.DM.growing_bbox[3])
        super(GrowTool, self).__init__(self.alevel_combo, 'ROI:',
                                       ULabel('z'), self.bboxz,
                                       ULabel('y'), self.bboxy,
                                       ULabel('x'), self.bboxx,
                                       'Lambda:', self.bbox_lamda,
                                       self.bbox_source, None,
                                       stretch=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        self.alevel_combo.currentIndexChanged.connect(self.on_change)
        self.bboxz.textEdited.connect(self.on_change)
        self.bboxy.textEdited.connect(self.on_change)
        self.bboxx.textEdited.connect(self.on_change)
        self.bbox_lamda.textChanged.connect(self.on_change)
        self.bbox_source.currentIndexChanged.connect(self.on_change)

    def on_change(self):
        bbox = [self.bboxz.value(), self.bboxy.value(), self.bboxx.value()]
        lamda = self.bbox_lamda.value()
        source = self.bbox_source.currentText()
        level = self.alevel_combo.currentIndex()

        if level == 1 and self.DM.svlabels is None:
            self.launcher.error.emit('SuperVoxels need to be created first')
            self.alevel_combo.setCurrentIndex(0)
            return

        if not self.DM.has_ds('data/{}'.format(source)):
            self.bbox_source.setCurrentIndex(0)
            errmsg = 'Dataset {} has to be created first'.format(source)
            QtWidgets.QMessageBox.critical(self, 'Error', errmsg)
            return

        self.DM.growing_bbox = [source, lamda, bbox, level]
