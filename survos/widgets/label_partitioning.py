

import numpy as np
import pandas as pd
from ..qt_compat import QtGui, QtCore

from .mpl_widgets import PerspectiveCanvas, MplCanvas
from .base import HWidgets, TComboBox, RoundedWidget, ColorButton, PLineEdit, \
                  SComboBox, CheckableLabels, SourceCombo

from .. import actions as ac

from ..plugins.base import Plugin
from ..core import DataModel, LabelManager, Launcher, LayerManager

import seaborn as sns
import logging as log


FEATURE_OPTIONS = [
    'Average Intensity',
    #'Median Intensity',
    'Sum Intensity',
    'Standard Devitation',
    'Variance',
    'Size (Area)',
    'Log10(Size) (Area)',
    'Size (Bounding Box)',
    'Depth (Bounding Box)',
    'Height (Bounding Box)',
    'Width (Bounding Box)',
    'Log10(Size) (Bounding Box)',
    'Size (Oriented Bounding Box)',
    '1st Axis (Oriented Bounding Box)',
    '2nd Axis (Oriented Bounding Box)',
    '3rd Axis (Oriented Bounding Box)',
    'Log10(Size) (Oriented Bounding Box)',
    'Position (X)',
    'Position (Y)',
    'Position (Z)'
]

FEATURE_TYPES = [
    'intensity',
    #'median',
    'sum',
    'std',
    'var',
    'area',
    'log_area',
    'size_bbox',
    'depth_bbox',
    'height_bbox',
    'width_bbox',
    'log_size_bbox',
    'size_ori_bbox',
    'depth_ori_bbox',
    'height_ori_bbox',
    'width_ori_bbox',
    'log_size_ori_bbox',
    'z_pos',
    'y_pos',
    'x_pos'
]

class LabelCanvas(PerspectiveCanvas):

    labelSelected = QtCore.pyqtSignal(int)

    def __init__(self):
        super(LabelCanvas, self).__init__(axisoff=False, autoscale=False)

        self.DM = DataModel.instance()
        self.DM.evmin_changed.connect(self.evmin_changed)
        self.DM.evmax_changed.connect(self.evmax_changed)
        self.DM.roi_changed.connect(self.initialize)
        self.slider.valueChanged.connect(self.on_value_changed)
        self.canvas.mpl_connect('button_release_event', self.on_select_object)
        self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

        self.data = None
        self.objects = None
        self.num_objects = None
        self.selected_object = None
        self.labels = None
        self.color_idx = None
        self.colors = None
        self.initialize()

    def initialize(self):
        self.data = '/data'
        if self.data is None:
            return False
        self.shape = self.DM.region_shape()
        self.vmin = self.DM.attr('/data', 'evmin')
        self.vmax = self.DM.attr('/data', 'evmax')
        self.slider.setMaximum(self.shape[0]-1)
        self.slider.setValue(self.shape[0]//2)
        self.on_value_changed(self.shape[0]//2)
        return True

    def on_value_changed(self, val):
        self.idx = val
        self.text_idx.setText(str(val))
        self.replot()

    def evmin_changed(self, vmin):
        self.vmin = vmin
        if len(self.ax.images):
            self.ax.images[0].set_clim(vmin=vmin)
        self.redraw()

    def evmax_changed(self, vmax):
        self.vmax = vmax
        if len(self.ax.images):
            self.ax.images[0].set_clim(vmax=vmax)
        self.redraw()

    def replot(self):
        self.ax.clear()
        if self.data is None:
            return
        curr_data = self.DM.load_slices(self.data, self.idx)[0]
        self.ax.imshow(curr_data, 'gray',
                       vmin=self.vmin, vmax=self.vmax)
        if self.objects:
            objs = self.DM.load_slices(self.objects, self.idx)[0]
            omax = objs.max()
            if omax > 0:
                self.ax.contour(objs, colors='#0000FF', linewidths=2,
                                levels=range(omax))

        if self.labels is not None:
            objs = self.DM.load_slices(self.objects, self.idx)[0]
            labels = self.labels[objs]
            for cidx in self.color_idx:
                color = self.colors[cidx]
                curr_obj = (labels == cidx)
                if np.unique(curr_obj).size > 0:
                    self.ax.contour(curr_obj, colors=color, linewidths=2)

        if self.selected_object is not None:
            objs = self.DM.load_slices(self.objects, self.idx)[0]
            mask = (objs == self.selected_object)
            if mask.any():
                self.ax.contour(mask, colors='#FF0000',
                                linewidths=2)
        self.ax.axis('image')
        self.redraw()

    def set_objects(self, objects, num_objects):
        self.objects = objects
        self.num_objects = num_objects
        self.selected_object = None
        self.replot()

    def on_select_object(self, ev):
        if ev.inaxes != self.ax or self.objects is None:
            return

        y = int(ev.ydata)
        x = int(ev.xdata)
        l = self.DM.load_slices(self.objects, self.idx)[0, y, x]
        if l > 0:
            self.selected_object = l
            self.replot()
            self.labelSelected.emit(l)

    def select_object(self, obj):
        self.selected_object = obj
        self.replot()

    def colour(self, labels, idx, colors):
        self.labels = np.r_[-1, labels]
        self.color_idx = idx
        self.colors = colors
        self.replot()

    def reset(self):
        self.labels = None
        self.objects = None
        self.num_objects = None
        self.selected_object = None
        self.replot()


class LabelExplorer(QtGui.QWidget):

    def __init__(self, label_canvas):
        super(LabelExplorer, self).__init__()

        self.label_canvas = label_canvas
        self.DM = DataModel.instance()
        self.LBLM = LabelManager.instance()

        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)

        self.level_combo = SComboBox()
        self.level_combo.setMinimumWidth(200)
        self.label_combo = CheckableLabels(-1)
        self.label_combo.setMinimumWidth(200)
        self.source_combo = SourceCombo()
        self.source_combo.setMinimumWidth(200)
        self.label_btn = QtGui.QPushButton('Label')
        vbox.addWidget(HWidgets(self.level_combo, self.label_combo,
                                self.source_combo, self.label_btn,
                                stretch=[1, 1, 1, 0]))

        self.mplcanvas = MplCanvas(axisoff=False, autoscale=True)
        self.label_rules = LabelRules()

        self.feature_combo = TComboBox('Select measure:', FEATURE_OPTIONS)
        self.kernels = TComboBox('Fit kernel:', ['gau', 'cos', 'biw', 'epa',
                                                'tri', 'triw'],
                                 selected=2)
        self.export_plot = QtGui.QPushButton('Export plot')
        self.export_stats = QtGui.QPushButton('Export Stats')
        vbox.addWidget(HWidgets(self.feature_combo, None,
                                self.kernels, self.export_plot, self.export_stats,
                                stretch=[0, 1, 0, 0, 0]))

        splitter = QtGui.QSplitter(0)
        vbox.addWidget(splitter)

        splitter.addWidget(self.mplcanvas)
        splitter.addWidget(self.label_rules)

        self.LBLM.levelLoaded.connect(self.on_level_added)
        self.LBLM.levelAdded.connect(self.on_level_added)
        self.LBLM.levelRemoved.connect(self.on_level_removed)

        self.level_combo.currentIndexChanged.connect(self.on_level_changed)
        self.label_btn.clicked.connect(self.on_label_clicked)
        self.feature_combo.currentIndexChanged.connect(self.on_feature_changed)
        self.label_canvas.labelSelected.connect(self.on_label_selected)
        self.export_plot.clicked.connect(self.on_export_plot)
        self.export_stats.clicked.connect(self.on_export_stats)
        self.kernels.currentIndexChanged.connect(self.on_kernel_changed)
        self.label_rules.computed.connect(self.on_rules_computed)
        self.label_rules.compute_others.connect(self.on_rules_computed)

        self.mplcanvas.canvas.mpl_connect('button_release_event',
                                           self.on_select_object)

        self.selected_label = None
        self.kernel = 'biw'
        self.feature = None
        self.colors = None
        self.color_idx = None
        self.labels = None
        self.objects = None
        self.computed_level = None
        self.levels = []

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
        i = self.levels.index(level)
        self.level_combo.removeItem(i)
        del self.levels[i]
        if self.computed_level is not None and level == self.computed_level:
            self.reset()

    def on_label_clicked(self):
        self.reset()

        level = self.levels[self.level_combo.currentIndex()]
        self.computed_level = level
        self.LBLM.save(level)
        labels = self.label_combo.getSelectedLabels()

        in_dset = self.LBLM.dataset(level)
        out_dset = 'objects/objects'
        source = self.source_combo.value()

        out_features = []
        for f in FEATURE_TYPES:
            out_features.append('objects/{}'.format(f))

        self.label_rules.initialize()

        self.DM.remove_dataset('objects/objects')
        self.DM.remove_dataset('objects/objlabels')

        Launcher.instance().run(ac.label_objects, dataset=in_dset, source=source,
                                out=out_dset, out_features=out_features,
                                labels=labels, caption='Labelling objects...',
                                cb=self.on_objects_labelled)

    def on_objects_labelled(self, params):
        objects, num_objects = params[:2]
        if objects is None:
            return
        self.objects = objects
        self.num_objects = num_objects
        self.label_canvas.set_objects(objects, num_objects)
        self.on_feature_changed()

    def on_feature_changed(self):
        feature_idx = self.feature_combo.currentIndex()
        stype = FEATURE_TYPES[feature_idx]
        self.show_stats(stype=stype)

    def show_stats(self, stype='intensity'):
        if stype == 'intensity':
            self.title = 'Average Intensity inside objects'
        elif stype == 'median':
            self.title = 'Median Intensity inside objects'
        elif stype == 'sum':
            self.title = 'Sum of Intensity inside objects'
        elif stype == 'std':
            self.title = 'Standard deviation of intensity inside objects'
        elif stype == 'var':
            self.title = 'Variance of intensity inside objects'
        elif stype == 'area':
            self.title = 'Area of objects (size in voxels)'
        elif stype == 'logarea':
            self.title = 'log10(Area) of objects (log size in voxels)'
        elif stype == 'bbox':
            self.title = 'Size of bounding box enclosing objects'
        elif stype == 'logbbox':
            self.title = 'log10(Size) of bounding box enclosing objects'
        elif stype == 'posx':
            self.title = 'Position (X) of the center of mass of the objects'
        elif stype == 'posy':
            self.title = 'Position (Y) of the center of mass of the objects'
        elif stype == 'posz':
            self.title = 'Position (Z) of the center of mass of the objects'

        self.feature = self.DM.load_ds('objects/{}'.format(stype))
        self.replot()

    def replot(self):
        self.mplcanvas.ax.clear()
        sns.distplot(self.feature, hist=False, kde=True, rug=False,
                     kde_kws={'lw': 3, 'shade': True, 'kernel': self.kernel},
                     ax=self.mplcanvas.ax)
        if self.labels is None:
            sns.rugplot(self.feature, ax=self.mplcanvas.ax)
        else:
            num_objects = self.num_objects
            for i in range(num_objects):
                l = self.labels[i]
                c = self.colors[l] if l >= 0 else '#0000FF'
                f = self.feature[i]
                self.mplcanvas.ax.axvline(f, 0, 0.05, lw=1, c=c)
        if self.selected_label is not None:
            f = self.feature[self.selected_label - 1]
            self.mplcanvas.ax.axvline(f, ymax=0.08, linewidth=3, color='#FF0000')
            self.mplcanvas.ax.set_title('Selectec value: {}'.format(f))
        self.mplcanvas.ax.set_xlabel(self.title)
        self.mplcanvas.redraw()

    def on_label_selected(self, label):
        self.selected_label = label
        self.replot()

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

        if self.labels is not None:
            total += 1

        data = np.zeros((self.num_objects, total), np.float32)

        for n, ftype in enumerate(features):
            feature = self.DM.load_ds('objects/{}'.format(ftype))
            data[:, n] = feature[:]

        if self.labels is not None:
            features += ['class']
            data[:, -1] = self.labels

        full_path = QtGui.QFileDialog.getSaveFileName(self, "Select output filename",
                                                      filter='*.csv')
        if full_path is not None and len(full_path) > 0:
            if not full_path.endswith('.csv'):
                full_path += '.csv'
            df = pd.DataFrame(data, columns=features)
            df.to_csv(full_path)

    def on_kernel_changed(self):
        self.kernel = self.kernels.currentText()
        self.replot()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape \
                and self.selected_label is not None:
            self.selected_label = None
            self.label_canvas.select_object(None)
            self.replot()

    def on_select_object(self, ev):
        if ev.inaxes != self.mplcanvas.ax or self.feature is None:
            return

        x = ev.xdata
        self.selected_label = np.abs(self.feature[:] - x).argmin() + 1
        self.label_canvas.select_object(self.selected_label)
        self.replot()

    def on_rules_computed(self):
        self.labels = self.DM.load_ds('objects/objlabels')
        self.color_idx = list(set(np.unique(self.labels)) - set([-1]))
        self.colors = {i:self.label_rules.labels[i].colorbutton.color
                       for i in self.color_idx}
        self.label_canvas.colour(self.labels, self.color_idx, self.colors)
        self.replot()

    def reset(self):
        self.label_canvas.reset()
        self.label_rules.reset()

        self.objects = None
        self.selected_label = None
        self.feature = None
        self.colors = None
        self.color_idx = None
        self.labels = None
        self.computed_level = None
        self.mplcanvas.ax.clear()
        self.mplcanvas.redraw()


class LabelRule(QtGui.QWidget):

    deleted = QtCore.pyqtSignal(int)

    def __init__(self, idx, parent=None):
        super(LabelRule, self).__init__(parent=parent)

        self.idx = idx
        vbox = vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)

        delbutton = QtGui.QPushButton('X')
        delbutton.setMaximumWidth(30)
        label = QtGui.QLabel('Rule {}'.format(idx+1))
        self.fcombo = TComboBox('Feature:', FEATURE_OPTIONS)
        self.opcombo = TComboBox('', ['>', '<'])
        self.threshold = PLineEdit(0.5, parse=float)

        vbox.addWidget(HWidgets(delbutton, label, None, self.fcombo,
                                None, self.opcombo, None, self.threshold,
                                stretch=[0, 0, 1, 0, 1, 0, 1, 0]))

        delbutton.clicked.connect(self.on_delete)

    def on_delete(self):
        self.deleted.emit(self.idx)


class LabelOption(RoundedWidget):

    computed = QtCore.pyqtSignal(int, str, str, list)
    deleted = QtCore.pyqtSignal(int)
    compute_others = QtCore.pyqtSignal(int)

    def __init__(self, idx, parent=None):
        super(LabelOption, self).__init__(parent=parent)

        self.idx = idx
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)

        delbutton = QtGui.QPushButton('X')
        delbutton.setMaximumWidth(30)
        self.colorbutton = ColorButton()
        self.label = PLineEdit('Label {}'.format(idx+1), parse=str)
        self.label.setText('Label {}'.format(idx+1))
        rulebutton = QtGui.QPushButton('Add new rule')
        bapply = QtGui.QPushButton('Apply')
        othersbutton = QtGui.QPushButton('Select Others')
        vbox.addWidget(HWidgets(delbutton, self.colorbutton, self.label, None,
                                rulebutton, bapply, othersbutton,
                                stretch=[0, 0, 0, 1, 0]))

        rulebutton.clicked.connect(self.add_rule)
        delbutton.clicked.connect(self.on_delete_label)
        bapply.clicked.connect(self.on_apply)
        othersbutton.clicked.connect(self.on_apply_others)

        self.rules = {}
        self.total_tules = 0

    def add_rule(self):
        new_rule = LabelRule(self.total_tules)
        new_rule.deleted.connect(self.on_rule_deleted)
        self.layout().addWidget(new_rule)
        self.rules[self.total_tules] = new_rule
        self.total_tules += 1

    def on_rule_deleted(self, idx):
        self.rules[idx].setParent(None)
        del self.rules[idx]

    def on_delete_label(self):
        self.deleted.emit(self.idx)

    def on_apply(self):
        name = self.label.value()
        color = self.colorbutton.color
        rules = []
        for rule in self.rules.values():
            idx = rule.fcombo.currentIndex()
            smaller = rule.opcombo.currentIndex()
            thresh = rule.threshold.value()
            rules += [(idx, smaller, thresh)]

        if len(rules) > 0:
            self.computed.emit(self.idx, name, color, rules)

    def on_apply_others(self):
        self.compute_others.emit(self.idx)


class LabelRules(QtGui.QWidget):

    computed = QtCore.pyqtSignal()
    compute_others = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(LabelRules, self).__init__(parent=parent)

        self.DM = DataModel.instance()
        self.setMinimumHeight(300)

        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)

        self.save_labels = QtGui.QPushButton('Save labels')
        self.level_combo = SComboBox()
        self.add_label = QtGui.QPushButton('Add new label')
        vbox.addWidget(HWidgets(self.level_combo, self.save_labels,
                                None, self.add_label, stretch=[0, 0, 1, 0]))

        groupbox = QtGui.QWidget()
        self.form = QtGui.QFormLayout()
        groupbox.setLayout(self.form)

        scroll = QtGui.QScrollArea()
        scroll.setWidget(groupbox)
        scroll.setWidgetResizable(True)
        vbox.addWidget(scroll)

        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.LBLM.levelLoaded.connect(self.on_level_added)
        self.LBLM.levelAdded.connect(self.on_level_added)
        self.LBLM.levelRemoved.connect(self.on_level_removed)
        self.add_label.clicked.connect(self.on_add_label)
        self.save_labels.clicked.connect(self.on_save_labels)

        self.label_count = 0
        self.labels = {}
        self.levels = []
        self.rules_computed = None
        self.initialized = None

    def initialize(self):
        self.initialized = True

    def on_level_added(self, level):
        self.levels.append(level)
        self.level_combo.addItem('Level {}'.format(level))
        self.level_combo.setCurrentIndex(level)

    def on_level_removed(self, level):
        idx = self.levels.index(level)
        self.level_combo.removeItem(idx)
        del self.levels[idx]

    def on_add_label(self):
        if self.initialized is None:
            return
        new_label = LabelOption(self.label_count)
        new_label.deleted.connect(self.on_delete_label)
        new_label.computed.connect(self.on_compute)
        new_label.compute_others.connect(self.on_compute_others)
        self.form.addRow(new_label)
        self.labels[self.label_count] = new_label
        self.label_count += 1

    def on_delete_label(self, idx):
        self.labels[idx].setParent(None)
        del self.labels[idx]

        if self.DM.has_ds('objects/objlabels'):
            data = self.DM.load_ds('objects/objlabels')
            data[data == idx] = -1
            self.DM.write_dataset('objects/objlabels', data)
        self.computed.emit()

    def on_compute(self, idx, name, color, rules):
        num_objects = self.DM.attr('objects/objects', 'num_objects')
        features = ['objects/{}'.format(f) for f in FEATURE_TYPES]
        out_ds = 'objects/objlabels'

        if not self.DM.has_ds(out_ds):
            self.DM.create_empty_dataset(out_ds, (num_objects,), np.int32)

        Launcher.instance().run(ac.apply_rules, features=features, label=idx,
                                num_objects=num_objects, rules=rules, out_ds=out_ds,
                                caption='Applying Rules', cb=self.on_rules_computed)

    def on_rules_computed(self, labels):
        self.rules_computed = True
        self.computed.emit()

    def on_compute_others(self, idx):
        num_objects = self.DM.attr('objects/objects', 'num_objects')
        data = self.DM.load_ds('objects/objlabels')
        if data is None:
            return
        data[data < 0] = idx
        self.DM.write_dataset('objects/objlabels', data)
        self.computed.emit()

    def on_save_labels(self):
        if self.rules_computed is None:
            return
        level = self.levels[self.level_combo.currentIndex()]
        quit_msg = "Are you sure you want to save new labels in [Level {}]?".format(level)
        reply = QtGui.QMessageBox.question(self, 'Message', quit_msg,
                                           QtGui.QMessageBox.Yes,
                                           QtGui.QMessageBox.No)

        if reply != QtGui.QMessageBox.Yes:
            return

        launcher = Launcher.instance()
        launcher.setup('Saving Labels on [Level {}]'.format(level))
        log.info('+ Loading objects')
        labels = self.DM.load_ds('objects/objlabels')
        objlabels = np.r_[-1, labels]
        current = self.DM.load_ds('objects/objects')

        log.info('+ Loading annotatins')
        target_ds = 'annotations/annotations{}'.format(level)
        target = self.DM.load_ds(target_ds)
        shift_label = target.max() + 1
        mask = (current > 0) & (objlabels[current] >= 0)
        target[mask] = objlabels[current[mask]] + shift_label
        self.DM.write_dataset(target_ds, target)

        log.info('+ Adding label widgets')
        labels = np.unique(labels)

        for l in labels:
            if l == -1:
                continue
            labelobj = self.labels[l]
            label = l + shift_label
            name = labelobj.label.value()
            color = labelobj.colorbutton.color
            self.LBLM.loadLabel(level, label, name, color, True, -1, -1)

        self.LBLM.save(level)
        self.LM.update()
        launcher.cleanup()

    def reset(self):
        self.label_count = 0
        for label in self.labels.values():
            label.setParent(None)
        self.labels = {}
        self.rules_computed = None
        self.initialized = None


class LabelSplitter(Plugin):

    name = 'Label Splitter'

    def __init__(self, data=None, ptype=Plugin.Widget):
        super(LabelSplitter, self).__init__(ptype=ptype)

        splitter = QtGui.QSplitter()
        self.addWidget(splitter)

        self.label_canvas = LabelCanvas()
        splitter.addWidget(self.label_canvas)

        self.label_explorer = LabelExplorer(self.label_canvas)
        splitter.addWidget(self.label_explorer)
        self.initialized = False

    def on_tab_changed(self, i):
        if self.initialized == False:
            self.initialized = self.label_canvas.initialize()
