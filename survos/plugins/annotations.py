from __future__ import unicode_literals    # at top of module

import numpy as np
import h5py as h5

from ..qt_compat import QtGui, QtCore, QtWidgets
from matplotlib.colors import ListedColormap

import os
import logging as log
from collections import OrderedDict

from .base import Plugin
from ..widgets import HWidgets, HEditLabel, HeaderLabel, PLineEdit, \
                      TComboBox, SubHeaderLabel, BLabel, RoundedWidget,\
                      ComboDialog, RCheckBox, HSize3D, SComboBox, ActionButton
from ..core import DataModel, LabelManager, LayerManager, Launcher
from .. import actions as ac
from six import iteritems


class AnnotationLayerLabel(QtWidgets.QWidget):

    def __init__(self, text, parent=None):
        super(AnnotationLayerLabel, self).__init__(parent=parent)

        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)

        spacing = QtWidgets.QPushButton('X')
        spacing.setMaximumWidth(25)
        spacing.setMaximumHeight(30)
        spacing.setMinimumHeight(30)
        spacing.setMinimumWidth(25)
        spacing.setStyleSheet('border-top-right-radius: 0px;'
                              'border-bottom-right-radius: 0px;'
                              'border-right: 2px solid red;')
        self.deleted = spacing.clicked

        hbox.addWidget(spacing)

        label = QtWidgets.QLabel(text)
        label.setMinimumHeight(30)
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        label.setStyleSheet('background-color: #6DC7C7; color: #006161;'
                            'padding-left: 10px;'
                            'border-top: 1px solid #fefefe;'
                            'border-bottom: 1px solid #fefefe;'
                            'font-weight: bold; font-size: 13px;')

        self.layout().setContentsMargins(0,0,0,0)
#        self.layout().setMargin(0)
        self.layout().setSpacing(0)

        hbox.addWidget(label, 1)

        self.chk = QtWidgets.QCheckBox()
        self.chk.setMinimumHeight(30)
        self.chk.setMaximumWidth(25)
        self.chk.setMaximumHeight(30)
        self.chk.setMinimumWidth(25)
        self.chk.setStyleSheet('background-color: #6DC7C7;'
                               'border-top-right-radius: 3px;'
                               'border-bottom-right-radius: 3px;'
                               'border: 1px solid #fefefe;'
                               'border-left: none')
        hbox.addWidget(self.chk)
        self.stateChanged = self.chk.stateChanged

    def setChecked(self, bol):
        self.chk.setChecked(bol)


class Level(RoundedWidget):

    parent_requested = QtCore.pyqtSignal(int, int)
    level_toggled = QtCore.pyqtSignal(int, bool)
    erase = QtCore.pyqtSignal(int)

    def __init__(self, name, level, ds, parent=None):
        super(Level, self).__init__(parent=parent, color=None, bg='#cde5e5')

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()

        self.name = name
        self.level = level
        self.dataset = ds

        self.setLayout(QtWidgets.QVBoxLayout())
        self.setContentsMargins(0,0,0,0)

        self.header = AnnotationLayerLabel(name)
        self.layout().addWidget(self.header)
        self.btn_addlabel = ActionButton('Add Label')
        self.btn_erase = ActionButton('Erase')
        self.btn_erase.setCheckable(True)
        self.layout().addWidget(HWidgets(None, self.btn_erase, self.btn_addlabel,
                                stretch=[1,0,0]))

        self.labels = OrderedDict()

        self.btn_addlabel.clicked.connect(self.on_add_label)
        self.btn_erase.clicked.connect(self.on_erase)
        self.header.deleted.connect(self.on_remove_level)

        self.LM.toggled.connect(self.toggle_level)
        self.header.stateChanged.connect(self.on_level_toggled)

    def num_labels(self):
        return len(self.labels)

    def toggle_level(self, name, level, val):
        if level == 'Annotations' and name == self.name:
            self.header.setChecked(val)

    def on_level_toggled(self, bol):
        self.level_toggled.emit(self.level, bol)

    def add_label(self, label, name):
        wlabel = HEditLabel(name, label)
        wlabel.selected.connect(self.on_selected)
        wlabel.removed.connect(self.on_remove_label)
        wlabel.nameChanged.connect(self.on_nameChanged)
        wlabel.colorChanged.connect(self.on_colorChanged)
        wlabel.visibilityChanged.connect(self.on_visibilityChanged)
        wlabel.parentRequested.connect(self.on_parentRequested)

        self.labels[label] = wlabel
        self.layout().addWidget(wlabel)

    def load_label(self, label, name, color, visible, parent_level, parent_label):

        wlabel = HEditLabel(name, label)
        wlabel.setName(name)
        wlabel.setColor(color)
        wlabel.setChecked(visible)

        wlabel.selected.connect(self.on_selected)
        wlabel.removed.connect(self.on_remove_label)
        wlabel.nameChanged.connect(self.on_nameChanged)
        wlabel.colorChanged.connect(self.on_colorChanged)
        wlabel.visibilityChanged.connect(self.on_visibilityChanged)
        wlabel.parentRequested.connect(self.on_parentRequested)

        self.labels[label] = wlabel
        self.layout().addWidget(wlabel)
        self.set_parent(label, parent_level, parent_label)

    def remove_label(self, label):
        if label in self.labels:
            self.labels[label].setParent(None)
            del self.labels[label]

    def setSelected(self, label):
        for k, lbl in iteritems(self.labels):
            lbl.setSelected(k == label)
        self.btn_erase.setChecked(label is not None and label < 0)

    def on_remove_level(self):
        self.LBLM.removeLevel(self.level)

    def on_add_label(self):
        self.LBLM.addLabel(self.level)

    def on_remove_label(self, label):
        answer = QtWidgets.QMessageBox.question(self,
                                                "Confirm deletion",
                                                "Do you want to permanently delete this annotation?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if answer == QtWidgets.QMessageBox.Yes:
            self.LBLM.removeLabel(self.level, label)
        else:
            log.info("+ Deleting annnotation was cancelled")

    def on_erase(self):
        self.erase.emit(self.level)

    def on_selected(self, label):
        self.LBLM.selectLabel(self.level, label)

    def on_nameChanged(self, label, name):
        self.LBLM.changeLabelName(self.level, label, str(name))

    def on_colorChanged(self, label, color):
        self.LBLM.changeLabelColor(self.level, label, str(color))

    def on_visibilityChanged(self, label, visible):
        self.LBLM.changeLabelVisibility(self.level, label, visible)

    def on_parentRequested(self, label):
        self.parent_requested.emit(self.level, label)

    def set_parent(self, label, parent_level, parent_label):
        if parent_label < 0:
            color = None
        else:
            color = self.LBLM.get(parent_level, parent_label).color
        self.labels[label].setParentColor(color)

    def update_parents(self, parent_level, parent_label):
        for label in self.labels.keys():
            obj = self.LBLM.get(self.level, label)
            if obj.parent_level == parent_level and \
                    obj.parent_label == parent_label:
                self.set_parent(label, parent_label)

    def visible(self, bol=None):
        if bol is None:
            return self.LM.get(self.name, 'Annotations').visible
        else:
            self.LM.setVisible(self.name, 'Annotations', bol)


class Annotations(Plugin):

    name = 'Annotations'

    def __init__(self):
        super(Annotations, self).__init__(ptype=Plugin.Plugin)

        # Relayout
        QtWidgets.QWidget().setLayout(self.layout)
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.setContentsMargins(0, 10, 0, 0)
        self.setLayout(vbox2)

        self.labels = {}
        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.launcher = Launcher.instance()

        self.levels = OrderedDict()

        #######
        vbox2.addWidget(HeaderLabel('Annotations'))

        dummy = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        dummy.setLayout(vbox)

        vbox2.addWidget(dummy, 1)

        vbox.addWidget(SubHeaderLabel('Annotation Levels'))

        self.level_background = PLineEdit(-1, parse=int)
        self.level_background.setMaximumWidth(40)
        self.btn_loadlevel = ActionButton('Load Level')
        self.btn_addlevel = ActionButton('Add Level')
        vbox.addWidget(HWidgets('Bg:', self.level_background,
                                self.btn_loadlevel, None, self.btn_addlevel,
                                stretch=[0, 0, 0, 1,0]))

        groupbox = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        groupbox.setLayout(self.form)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(groupbox)
        scroll.setWidgetResizable(True)
        vbox.addWidget(scroll, 1)

        ###############################################
        vbox.addWidget(SubHeaderLabel('Refine Annotation Label'))
        self.combo_refine = TComboBox('', ['dilation', 'erosion', 'opening',
                                           'closing', 'fill_holes'])
        self.refine_radius = PLineEdit(1, parse=int)
        self.refine_btn = ActionButton('Refine')
        self.slide_combo = SComboBox()
        self.slide_combo.addItem('This slice')
        self.slide_combo.addItem('All slices (2D)')
        self.slide_combo.addItem('Whole volume (3D)')
        vbox.addWidget(HWidgets(self.combo_refine, 'Radius:', self.refine_radius,
                                None, self.slide_combo, stretch=[1, 0, 0, 1, 0]))
        vbox.addWidget(HWidgets(None, self.refine_btn, stretch=[1, 0]))

        ### Signals
        self.btn_addlevel.clicked.connect(self.LBLM.addLevel)
        self.btn_loadlevel.clicked.connect(self.on_load_external_level)
        #self.previous_tool.clicked.connect(self.on_load_previous_tool_level)
        self.refine_btn.clicked.connect(self.refine_level)

        self.LBLM.levelAdded.connect(self.on_add_level)
        self.LBLM.levelLoaded.connect(self.on_load_level)
        self.LBLM.levelRemoved.connect(self.on_remove_level)
        self.LBLM.saveLevel.connect(self.save_labels)

        self.LBLM.labelAdded.connect(self.on_add_label)
        self.LBLM.labelRemoved.connect(self.on_remove_label)
        self.LBLM.labelLoaded.connect(self.on_load_label)
        self.LBLM.labelSelected.connect(self.on_select_label)
        self.LBLM.labelNameChanged.connect(self.on_label_name)
        self.LBLM.labelColorChanged.connect(self.on_label_color)
        self.LBLM.labelVisibilityChanged.connect(self.on_label_visibility)
        self.LBLM.labelParentChanged.connect(self.on_label_parent_changed)

    def refine_level(self):
        if self.DM.gtselected is None:
            QtWidgets.QMessageBox.critical(self, 'Error', 'No label selected')
            return

        level = self.DM.gtselected['level']
        label = self.DM.gtselected['label']
        refine_method = self.combo_refine.currentText()
        refine_radius = self.refine_radius.value()
        rslide = [self.DM.current_idx, '2D', '3D'][self.slide_combo.currentIndex()]

        self.launcher.run(ac.refine_label, data=level, label=label, slide=rslide,
                          method=refine_method, radius=refine_radius,
                          caption='Refining annotations..', cb=self.on_refined)

    def on_refined(self, params):
        if params is None:
            return
        indexes, values = params
        if len(indexes) == 0 or len(values) == 0:
            return

        zmin, ymin, xmin = indexes.min(0)
        zmax, ymax, xmax = indexes.max(0)

        indexes[:, 0] -= zmin
        indexes[:, 1] -= ymin
        indexes[:, 2] -= xmin

        slice_z = slice(zmin, zmax+1)
        slice_y = slice(ymin, ymax+1)
        slice_x = slice(xmin, xmax+1)

        level = self.DM.gtselected['level']
        self.DM.last_changes.append((level, (slice_z, slice_y, slice_x),
                                indexes, values, False))
        self.LM.update()

    def on_add_level(self, idx, ds):
        assert idx not in self.levels

        success = self.create_level_ds(ds)
        if not success:
            errmsg = 'Dataset "{}" already exists'.format(ds)
            QtWidgets.QMessageBox.critical(self, 'Error', errmsg)
            return
        else:
            self.DM.set_attrs(ds, dict(active=True, levelid=idx))

        name = u'Level {}'.format(idx)
        level = Level(name, idx, ds)
        level.level_toggled.connect(self.on_level_toggled)
        level.erase.connect(self.on_erase_clicked)
        level.parent_requested.connect(self.on_request_parent)

        self.levels[idx] = level
        self.form.addRow(level)

        cmap = ListedColormap(['#000000'])
        self.LM.addLayer(ds, name, 'Annotations', cmap=cmap,
                         background=-1, visible=False, vmin=0, vmax=1)

        self.save_labels(idx)
        log.info('+ Ready.')
        return ds

    def on_load_level(self, idx, ds):
        assert idx not in self.levels, "Error, level ID already exists"

        name = u'Level {}'.format(idx)
        level = Level(name, idx, ds)
        level.level_toggled.connect(self.on_level_toggled)
        level.erase.connect(self.on_erase_clicked)
        level.parent_requested.connect(self.on_request_parent)

        self.levels[idx] = level
        self.form.addRow(level)

        cmap = ListedColormap(['#000000'])
        self.LM.addLayer(ds, name, 'Annotations', cmap=cmap,
                         background=-1, visible=False, vmin=0, vmax=1)
        log.info('+ Loading [Level {}] -- "{}"'.format(idx, ds))
        return ds

    def on_load_external_level(self):
        self.launcher.setup("Importing annotation level")
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Select input source",
                                                 filter='*.h5 *.hdf5 *.h5.bak')
        if isinstance(path, tuple):
            path = path[0]
        if path is None or len(path) == 0:
            self.launcher.cleanup()
            return

        dataset = None

        p1 = os.path.basename(path)
        p2 = os.path.dirname(path)
        p3 = os.path.basename(p2)
        p4 = os.path.realpath(os.path.dirname(p2))
        p5 = os.path.realpath(self.DM.wspath)

        if p4 == p5 and p3 == 'annotations' and p1.endswith('.h5'):
            log.info('+ Attempting to re-activate a previous level')
            dataset = 'annotations/{}'.format(p1[:-3])
            attrs = self.DM.attrs(dataset)

            if 'levelid' in attrs and 'label' in attrs and 'parent_levels' in attrs \
                    and 'parent_labels' in attrs and 'colors' in attrs \
                    and 'names' in attrs and 'visible' in attrs and 'active' in attrs \
                    and (len(attrs['label']) == len(attrs['parent_levels']) \
                         == len(attrs['colors']) == len(attrs['parent_labels']) \
                         == len(attrs['names']) == len(attrs['visible'])):
                if attrs['active'] == False:
                    levels = self.LBLM.levels()
                    if attrs['levelid'] in levels:
                        levelid = max(levels) + 1
                        self.DM.set_attrs(dataset, dict(levelid=levelid))
                    else:
                        levelid = attrs['levelid']
                    self.DM.set_attrs(dataset, dict(active=True))
                    last_level, ds = self.LBLM.loadLevel(levelid, dataset)
                    labels = self.DM.attr(dataset, 'label')
                else:
                    errmsg = 'Selected level is already loaded'
                    QtWidgets.QMessageBox.critical(self, 'Error', errmsg)
                    return self.launcher.cleanup()
            else:
                log.info('    * Not possible to re-activate')


        if path.endswith('.h5') or path.endswith('.h5.bak'):
            av = self.DM.available_hdf5_datasets(path, dtype=np.int16)
            if len(av) == 0:
                self.launcher.show_error('No annotation level found')
                return self.launcher.cleanup()
            selected, accepted = ComboDialog.getOption(av)
            if accepted == QtWidgets.QDialog.Rejected:
                return
            dataset = selected

            background = self.level_background.value()

            with h5.File(path, 'r') as f:
                shape = f[dataset].shape
            dshape = self.DM.data_shape
            rshape = self.DM.region_shape()

            if shape != dshape and shape != rshape:
                errmsg = 'Annotation shape {} do not match with either loaded dataset {}' \
                         ' or selected ROI {}'.format(shape, dshape, rshape)
                self.launcher.show_error(errmsg)
                self.launcher.cleanup()
                return

            last_level, ds = self.LBLM.addLevel()

            with h5.File(path, 'r') as f:
                data = f[dataset][:]

            if background != -1:
                data[data == background] = -1
            if data.dtype != np.int32:
                data = data.astype(np.int32)

            log.info('+ Writing labels to disk')
            if shape == dshape:
                self.DM.write_dataset(ds, data)
            else:
                self.DM.write_slices(ds, data)

            labels = set(np.unique(data)) - set([-1])

        log.info('+ Populating widgets')
        for label in labels:
            self.LBLM.loadLabel(last_level, label, 'Label {}'.format(label),
                                '#000000', True, -1, -1)
        self.on_level_toggled(last_level, True)
        self.save_labels(last_level)
        log.info('+ Ready.')
        self.launcher.cleanup()

    def on_remove_level(self, idx, ds, force):
        visible = self.levels[idx].visible()

        log.info('+ Removing [Level {}]'.format(idx))
        self.LM.remove(self.levels[idx].name, 'Annotations')

        if self.DM.gtselected is not None and self.DM.gtselected['levelidx'] == idx:
            self.DM.gtselected = None

        log.info('+ Clearing annotations [Level {}]'.format(idx))
        self.remove_level_ds(idx, ds, force)

        log.info('+ Removing widgets')
        self.levels[idx].setParent(None)
        del self.levels[idx]

        if visible:
            self.LM.update()

        log.info('+ Ready.')

    def on_level_toggled(self, level, bol):
        for lobj in self.levels.values():
            if lobj.level == level and not bol:
                lobj.setSelected(None)

        if not bol and self.DM.gtselected is not None and \
                self.DM.gtselected['levelidx'] == level:
            self.DM.gtselected = None

        self.levels[level].visible(bol)
        self.LM.update()

    def on_erase_clicked(self, level):
        self.on_select_label(level, self.LBLM.dataset(level), -1)

    def on_load_label(self, level, dataset, label, name, color, visible,
                      parent_level, parent_label):
        log.info('  * Loading annotation [{}] for [Level {}]'.format(name, level))
        self.levels[level].load_label(label, name, color, visible, parent_level, parent_label)
        self.update_level_colormap(level)

    def on_add_label(self, level, dataset, label, name):
        self.levels[level].add_label(label, name)
        self.save_labels(level)
        self.update_level_colormap(level)

        if self.levels[level].visible():
            self.LM.update()
        log.info('+ Ready.')

    def on_remove_label(self, level, dataset, labelobj):
        label = labelobj.idx
        label_name = labelobj.name

        if self.DM.gtselected is not None:
            slevel = self.DM.gtselected['levelidx']
            slabel = self.DM.gtselected['label']
            if slevel == level:
                if slabel == label:
                    self.DM.gtselected = None

        self.launcher.setup('Removing label [{}] form [Level {}]'
                            .format(label_name, level))
        log.info('+ Loading annotation data into memory')
        data = self.DM.load_ds(dataset)
        data[data == label] = -1

        log.info('+ Writing changes to disk')
        self.DM.write_dataset(dataset, data)

        log.info('+ Removing widgets')
        self.levels[level].remove_label(label)
        self.save_labels(level)
        self.update_level_colormap(level)

        self.LM.update()

        self.launcher.cleanup()

    def on_select_label(self, level, dataset, label):
        for idx, lobj in iteritems(self.levels):
            if idx == level:
                lobj.setSelected(label)
                self.LM.setVisible(lobj.name, 'Annotations', True)
            else:
                lobj.setSelected(None)

        self.select_label(level, dataset, label)
        self.LM.update()

    def select_label(self, level, dataset, label):

        if level is None:
            self.DM.gtselected = None
        elif  self.DM.gtselected is None or self.DM.gtselected['levelidx'] != level:
            parent_level = parent_label = None
            if label >= 0:
                current = self.LBLM.get(level, label)
                if current.parent_level is not None and current.parent_label >= 0:
                    parent_level = current.parent_level
                    parent_label = current.parent_label

            self.DM.gtselected = {
                'levelidx' : level,
                'level' : self.LBLM.dataset(level),
                'label' : label,
                'parent_level' : parent_level,
                'parent_label' : parent_label
            }
            lobj = self.LM.get(self.levels[level].name, 'Annotations')
            lobj.data = self.LBLM.dataset(level)
        elif self.DM.gtselected['levelidx'] == level:
            parent_level = parent_label = None
            if label >= 0:
                current = self.LBLM.get(level, label)
                if current.parent_level is not None and current.parent_label >= 0:
                    parent_level = current.parent_level
                    parent_label = current.parent_label
            self.DM.gtselected['label'] = label
            self.DM.gtselected['parent_level'] = parent_level
            self.DM.gtselected['parent_label'] = parent_label

        log.info('+ Ready.')

    def on_label_name(self, level, label, name):
        self.save_labels(level)

    def on_label_color(self, level, label, name):
        self.save_labels(level)
        self.update_level_colormap(level)
        for level in self.levels.values():
            level.update_parents(level, label)
        self.LM.update()

    def on_label_visibility(self, level, label, name):
        self.save_labels(level)

    def update_level_colormap(self, level):
        colors = self.LBLM.colors(level)
        idxs = self.LBLM.idxs(level)
        maxid = 0 if len(idxs) == 0 else max(idxs)+1
        cmaplist = ['#000000'] * maxid
        for idx, color in zip(idxs, colors):
            cmaplist[idx] = color
        newcmap = ListedColormap(cmaplist)
        layer = self.LM.get(self.levels[level].name, 'Annotations')
        layer.cmap = newcmap
        layer.vmin = 0
        layer.vmax = maxid

    def save_labels(self, level):
        log.info('+ Updating labels for [Level {}]'.format(level))
        dataset = self.levels[level].dataset
        self.DM.set_attrs(dataset, dict(label   = self.LBLM.idxs(level),
                                        names   = self.LBLM.names(level),
                                        colors  = self.LBLM.colors(level),
                                        visible = list(map(int, self.LBLM.visibility(level))),
                                        parent_levels = self.LBLM.parent_levels(level),
                                        parent_labels = self.LBLM.parent_labels(level)))

    def create_level_ds(self, ds):
        log.info('### Creating annotations "{}" ###'.format(ds))
        return ds, self.DM.create_empty_dataset(ds, self.DM.data_shape,
                                                dtype=np.int16)

    def remove_level_ds(self, idx, ds, force):
        if not force:
            quit_msg = "Do you want to permanently remove the data from [Level {}]?".format(idx)
            wipe = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                              QtWidgets.QMessageBox.Yes,
                                              QtWidgets.QMessageBox.No)
            if wipe == QtWidgets.QMessageBox.Yes:
                log.info('### Removing [Level {}] annotations ###'.format(idx))
                self.DM.remove_dataset(ds)
            else:
                log.info('### Disabling [Level {}] annotations ###'.format(idx))
                self.DM.set_attrs(ds, dict(active=False))
        else: # Force removal of the dataset - only used when appending data to classifier
            log.info('### Removing [Level {}] annotations ###'.format(idx))
            self.DM.remove_dataset(ds)

    def on_request_parent(self, level, label):
        if level == 0:
            return

        current_index = list(self.levels.keys()).index(level)
        if current_index > 0:
            parent_level = list(self.levels.keys())[current_index-1]
            dataset = self.LBLM.dataset(level)
            parent_dataset = self.LBLM.dataset(parent_level)
        else:
            return

        labels = self.LBLM.labels(parent_level)
        if len(labels) == 0:
            return
        
        options = ['None'] + [u'Level {}/{}'.format(parent_level, l.name) for l in labels]
        option, result = ComboDialog.getOptionIdx(options)

        if not result:
            return

        if option == 0:
            selected = None
            parent_label = -1
        else:
            selected = labels[option - 1]
            parent_label = selected.idx

        if selected is None:
            if self.DM.gtselected is not None \
                    and self.DM.gtselected['levelidx'] == level \
                    and self.DM.gtselected['label'] == label:
                self.DM.gtselected['parent_level'] = None
            self.LBLM.setLabelParent(level, label, parent_level, parent_label)
            #self.levels[level].set_parent(label, parent_label)
            self.save_labels(level)
            return

        self.launcher.setup('Assigning parent to [{}] of [Level {}]'
                            .format(self.LBLM.labels(level)[label].name, level))

        dparent = self.DM.load_ds(parent_dataset)
        dcurrent = self.DM.load_ds(dataset)

        mparent = (dparent == parent_label)
        mcurrent = (dcurrent == label)
        mintersection = mcurrent & mparent
        if not (mcurrent == mintersection).all():
            err_msg = "Some annotations of the label will be lost, do you wish to continue?"
            ans = QtWidgets.QMessageBox.question(self, "Data will be lost", err_msg,
                                             QtWidgets.QMessageBox.Yes,
                                             QtWidgets.QMessageBox.No)
            if ans == QtWidgets.QMessageBox.No:
                return

            dcurrent[mcurrent] = -1
            dcurrent[mintersection] = label
            self.DM.write_dataset(dataset, dcurrent)

            slices = slice(None), slice(None), slice(None)
            indexes = np.column_stack(np.where(mcurrent))
            self.DM.last_changes.append((dataset, slices, indexes, label))

        self.LBLM.setLabelParent(level, label, parent_level, parent_label)
        self.launcher.cleanup()

    def on_label_parent_changed(self, level, dataset, label, parent_level, parent_label):
        if self.DM.gtselected is not None \
                and self.DM.gtselected['levelidx'] == level \
                and self.DM.gtselected['label'] == label:

            self.DM.gtselected['parent_level'] = parent_level
            self.DM.gtselected['parent_label'] = parent_label

        self.levels[level].set_parent(label, parent_level, parent_label)
        self.save_labels(level)
        self.LM.update()
