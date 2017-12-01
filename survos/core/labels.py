
import numpy as np
from ..qt_compat import QtGui, QtCore
from collections import OrderedDict

import logging as log

from .model import DataModel
from .launcher import Launcher
from .singleton import Singleton

Axial = DataModel.instance().Axial
Sagittal = DataModel.instance().Sagittal
Coronal = DataModel.instance().Coronal


class Label(object):

    def __init__(self, name, idx, color='#000000', visible=True,
                 parent_level=-1, parent_label=-1):
        self.name = name
        self.idx = idx
        self.color = color
        self.visible = visible
        self.parent_level = parent_level
        self.parent_label = parent_label

@Singleton
class LabelManager(QtCore.QObject):

    levelAdded = QtCore.pyqtSignal(int, str)
    levelLoaded = QtCore.pyqtSignal(int, str)
    levelRemoved = QtCore.pyqtSignal(int, str)
    saveLevel = QtCore.pyqtSignal(int, str)

    labelAdded = QtCore.pyqtSignal(int, str, int, str)
    labelLoaded = QtCore.pyqtSignal(int, str, int, str, str, bool, int, int)
    labelRemoved = QtCore.pyqtSignal(int, str, object)
    labelUpdated = QtCore.pyqtSignal(int, str)
    labelSelected = QtCore.pyqtSignal(int, str, int)
    labelNameChanged = QtCore.pyqtSignal(int, str, int, str)
    labelColorChanged = QtCore.pyqtSignal(int, str, int, str)
    labelVisibilityChanged = QtCore.pyqtSignal(int, str, int, bool)
    labelParentChanged = QtCore.pyqtSignal(int, str, int, int, int)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.DM = DataModel.instance()
        self._levels = OrderedDict()
        self._counts = OrderedDict()
        self._datasets = OrderedDict()
        self.next_level = 0

    def len(self, level=None):
        if level is None:
            return len(self._levels)
        else:
            return len(self._levels[level])

    def levels(self):
        return self._levels.keys()

    def labels(self, level):
        return self._levels[level].values()

    def dataset(self, level):
        return self._datasets[level]

    def idxs(self, level):
        return [label.idx for label in self._levels[level].values()]

    def colors(self, level):
        return [label.color for label in self._levels[level].values()]

    def names(self, level):
        return [label.name for label in self._levels[level].values()]

    def parent_levels(self, level):
        return [label.parent_level for label in self._levels[level].values()]

    def parent_labels(self, level):
        return [label.parent_label for label in self._levels[level].values()]

    def visibility(self, level):
        return [label.visible for label in self._levels[level].values()]

    def foundLevel(self, levelidx):
        if levelidx >= self.next_level:
            self.next_level = levelidx + 1

    def loadLevel(self, levelidx, dataset):
        if levelidx in self._levels:
            log.error('  --- Annotations [{}] has same ID than [{}]. Skipping.'.format(
                dataset, self.dataset(levelidx)
            ))
            self.DM.set_attrs(dataset, dict(active=False))
            return
        self._levels[levelidx] = OrderedDict()
        self._counts[levelidx] = 0
        self._datasets[levelidx] = dataset
        self.levelLoaded.emit(levelidx, dataset)
        if levelidx >= self.next_level:
            self.next_level = levelidx + 1

        attrs = self.DM.attrs(dataset)
        label = attrs['label']
        names = attrs['names']
        colors = attrs['colors']
        visible = attrs['visible']
        parent_levels = attrs['parent_levels']
        parent_labels = attrs['parent_labels']

        if label is not None and len(label) > 0:
            for l, n, c, v, ple, pla in zip(label, names, colors, visible,
                                            parent_levels, parent_labels):
                self.loadLabel(levelidx, l, n, c, v > 0, ple, pla)

        return levelidx, dataset

    def addLevel(self):
        levelidx = self.next_level
        dataset = 'annotations/annotations{}'.format(levelidx)
        self._levels[levelidx] = OrderedDict()
        self._counts[levelidx] = 0
        self._datasets[levelidx] = dataset
        self.levelAdded.emit(levelidx, dataset)
        self.next_level += 1
        return levelidx, dataset

    def removeLevel(self, level):
        dataset = self._datasets[level]
        del self._levels[level]
        del self._counts[level]
        del self._datasets[level]
        self.levelRemoved.emit(level, dataset)

    def addLabel(self, level):
        idx = self._counts[level]
        name = 'Label {}'.format(idx)
        self._levels[level][idx] = Label(name, idx)
        self.labelAdded.emit(level, self._datasets[level], idx, name)
        self._counts[level] += 1

    def loadLabel(self, level, label, name, color, visible, parent_level, parent_label):
        self._levels[level][label] = Label(name, label, color.decode('UTF-8'), visible, parent_level, parent_label)
        self.labelLoaded.emit(level, self._datasets[level], label, name.decode('UTF-8'), color.decode('UTF-8'),
                              visible, parent_level, parent_label)
        if label >= self._counts[level]:
            self._counts[level] = label+1

    def get(self, level, label):
        return self._levels[level][label]

    def removeLabel(self, level, label):
        if label in self._levels[level]:
            labelobj = self._levels[level][label]
            del self._levels[level][label]
            self.labelRemoved.emit(level, self._datasets[level], labelobj)

    def isin(self, level, label):
        return label in self._levels[level]

    def index(self, level, label):
        return list(self._levels[level].keys()).index(name)

    def clear(self, level):
        self._levels[level].clear()

    def setLabelParent(self, level, label, parent_level, parent_label):
        self._levels[level][label].parent_level = parent_level
        self._levels[level][label].parent_label = parent_label
        self.labelParentChanged.emit(level, self._datasets[level], label,
                                     parent_level, parent_label)

    def changeLabelName(self, level, label, name):
        self._levels[level][label].name = name
        self.labelNameChanged.emit(level, self._datasets[level], label, name)

    def changeLabelColor(self, level, label, color):
        self._levels[level][label].color = color
        self.labelColorChanged.emit(level, self._datasets[level], label, color)

    def changeLabelVisibility(self, level, label, visible):
        self._levels[level][label].visible = visible
        self.labelVisibilityChanged.emit(level, self._datasets[level], label, visible)

    def selectLabel(self, level, label):
        self.labelSelected.emit(level, self._datasets[level], label)

    def save(self, level):
        self.saveLevel.emit(level, self._datasets[level])

    def saveAll(self):
        for level in self.levels():
            self.saveLevel.emit(level, self._datasets[level])
