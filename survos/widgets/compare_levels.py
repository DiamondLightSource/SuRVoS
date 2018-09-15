

import numpy as np
import pandas as pd
import seaborn as sns

from ..qt_compat import QtGui, QtCore, QtWidgets

from .. import actions as ac
from ..plugins import Plugin
from ..core import LabelManager, Launcher

from .mpl_widgets import MplCanvas
from .base import HeaderLabel, SComboBox, RoundedWidget, \
    TComboBox, HWidgets, SubHeaderLabel, ColorButton, RCheckBox


class CompareLevel(QtWidgets.QWidget):

    def __init__(self, name, parent=None):
        super(CompareLevel, self).__init__(parent=parent)

        self.LBLM = LabelManager.instance()

        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(vbox)
        vbox.addWidget(HeaderLabel('Level {}'.format(name)))

        self.level_combo = SComboBox()
        self.level_combo.addItem('Select level')
        self.level_combo.setMinimumWidth(200)
        vbox.addWidget(self.level_combo)

        self.container = QtWidgets.QWidget()
        vbox2 = QtWidgets.QVBoxLayout(self.container)
        vbox2.setAlignment(QtCore.Qt.AlignTop)
        vbox.addWidget(self.container, 1)

        self.LBLM.levelLoaded.connect(self.on_level_added)
        self.LBLM.levelAdded.connect(self.on_level_added)
        self.LBLM.levelRemoved.connect(self.on_level_removed)

        self.LBLM.labelAdded.connect(self.on_label_added)
        self.LBLM.labelRemoved.connect(self.on_label_removed)
        self.LBLM.labelNameChanged.connect(self.on_label_name)
        self.LBLM.labelColorChanged.connect(self.on_label_color)

        self.levels = [None]
        self.labels = {}
        self.level_selected = -1

        self.level_combo.currentIndexChanged.connect(self.on_level_changed)

    def on_level_added(self, level):
        self.levels.append(level)
        self.level_combo.addItem('Level {}'.format(level))
        if level == 0:
            self.on_level_changed(0)

    def on_level_removed(self, level):
        i = self.levels.index(level)
        self.level_combo.removeItem(i)
        del self.levels[i]

    def on_level_changed(self, level):
        for label in self.labels.values():
            label.setParent(None)
        self.labels.clear()

        self.level_selected = self.levels[level]
        try:
            for label in self.LBLM.labels(self.levels[level]):
                widget = CompareLabel(label.idx, label.name, label.color)
                self.labels[label.idx] = widget
                self.container.layout().addWidget(widget)
        except:
            pass

    def on_label_added(self, level, level_name, label, label_name):
        if level != self.level_selected or label in self.labels:
            return
        label_color = self.LBLM.get(level, label).color
        widget = CompareLabel(label, label_name, label_color)
        self.labels[label] = widget
        self.container.layout().addWidget(widget)

    def on_label_removed(self, level, level_name, label):
        if level != self.level_selected:
            return
        if label.idx in self.labels:
            self.labels[label.idx].setParent(None)
            del self.labels[label.idx]

    def on_label_name(self, level, level_name, label, label_name):
        if level != self.level_selected:
            return
        if label in self.labels:
            self.labels[label].setName(label_name)

    def on_label_color(self, level, level_name, label, label_color):
        if level != self.level_selected:
            return
        if label in self.labels:
            self.labels[label].setColor(label_color)

    def value(self):
        labels = {l.source: l.target for l in self.labels.values()}
        return dict(
            level=self.level_selected, labels=labels
        )


class CompareLabel(RoundedWidget):

    def __init__(self, idx, name, color, parent=None):
        super(CompareLabel, self).__init__(
            parent=parent, color=None, bg='#cde5e5', width=0)
        self.idx = idx
        self.name = name

        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)

        self.color = ColorButton(color=color, clickable=False)
        hbox.addWidget(self.color)

        self.label = QtWidgets.QLabel(name)
        hbox.addWidget(self.label, 1)

        self.combo = TComboBox('Target ID', ['Ignore'] + list(range(25)),
                               parse=int, default=-1)
        hbox.addWidget(self.combo)

    @property
    def source(self):
        return self.idx

    @property
    def target(self):
        return self.combo.value()

    def setName(self, name):
        self.label.setText(name)

    def setColor(self, color):
        self.color.setColor(color)


class ScoreWidget(RoundedWidget):

    def __init__(self, name, score, parent=None):
        super(ScoreWidget, self).__init__(
            parent=parent, color=None, bg='#cde5e5', width=0)
        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)
        slabel = QtWidgets.QLabel('{0:.4f}'.format(score))
        slabel.setFixedWidth(80)
        slabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        hbox.addWidget(HWidgets(name, None, slabel,
                                sctretch=[False, True, False]))


class ComparisonResult(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(ComparisonResult, self).__init__(parent=parent)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        self.setMaximumHeight(400)
        vbox.addWidget(HeaderLabel('Results'))

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)

        self.scores = QtWidgets.QWidget()
        self.scores.setMinimumWidth(300)
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.setAlignment(QtCore.Qt.AlignTop)
        self.scores.setLayout(vbox2)

        self.mplcanvas = MplCanvas(axisoff=False)

        hbox.addWidget(self.scores)
        hbox.addWidget(self.mplcanvas, 1)

    def setResults(self, results):
        scores = []
        if 'dice' in results:
            scores.append(('Dice Coefficient', results['dice']))
        if 'jacc' in results:
            scores.append(('Jaccard Index', results['jacc']))
        if 'cohen' in results:
            scores.append(('Cohen\'s Kappa', results['cohen']))

        countsA = results['countsA']
        countsB = results['countsB']
        countsAB = results['countsOverlap']
        indexes = results['indexes']

        #
        # SCORES
        #

        for i in reversed(range(self.scores.layout().count())):
            self.scores.layout().itemAt(i).widget().setParent(None)

        for name, score in scores:
            widget = ScoreWidget(name, score)
            self.scores.layout().addWidget(widget)

        #
        # COUNTS A
        #

        self.mplcanvas.fig.clear()
        self.mplcanvas.ax.clear()

        t = list(indexes - {-1})
        n = len(t)

        for i, idx in enumerate(t):
            x = range(3)
            y = [countsA[idx + 1], countsB[idx + 1], countsAB[idx + 1]]

            ax = self.mplcanvas.fig.add_subplot(1, n, i + 1)
            ax.bar(x, y, color=['#2aa22a', '#3399ff', '#ff5050'])
            ax.set_xticks([j + 0.35 for j in x])
            ax.set_xticklabels(['Level A', 'Level B', 'Overlap'])
            ax.set_title('Target {}'.format(idx))
            ax.set_ylabel('Volume area ([0,1] ratio)')

        sns.despine(self.mplcanvas.fig)
        self.mplcanvas.redraw()


class QuantitativeAnalysis(Plugin):
    name = 'Quantitative Analysis'

    def __init__(self, ptype=Plugin.Widget):
        super(QuantitativeAnalysis, self).__init__(ptype=ptype)

        self.LBLM = LabelManager.instance()

        self.layout.parent = None
        vbox = QtWidgets.QVBoxLayout(self)
        self.layout = vbox
        splitter = QtWidgets.QSplitter()

        self.level_widget1 = CompareLevel('A')
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.level_widget1)
        scroll.setWidgetResizable(True)
        splitter.addWidget(scroll)

        self.level_widget2 = CompareLevel('B')
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.level_widget2)
        scroll.setWidgetResizable(True)
        splitter.addWidget(scroll)

        self.results = ComparisonResult()

        vbox.addWidget(splitter, 1)
        self.dice = RCheckBox('Dice Coefficient')
        self.jacc = RCheckBox('Jaccard Score')
        self.jacc.setChecked(False)
        self.cohen = RCheckBox('Cohen\'s Kappa')
        self.cohen.setChecked(False)
        button = QtWidgets.QPushButton('Calculate Segmentation Coefficients')
        vbox.addWidget(HWidgets(None, self.dice, self.jacc,
                                self.cohen, button, None,
                                stretch=[True, False, False, False, False, False, True]))
        vbox.addWidget(self.results)

        button.clicked.connect(self.on_compare_clicked)

    def on_compare_clicked(self):
        l1 = self.level_widget1.value()
        l2 = self.level_widget2.value()

        launcher = Launcher.instance()

        if l1['level'] is None or l2['level'] is None:
            launcher.show_error('Level A or Level B is not selected')
            return

        levelA = self.LBLM.dataset(l1['level'])
        levelB = self.LBLM.dataset(l2['level'])
        labelsA = l1['labels']
        labelsB = l2['labels']

        launcher.run(ac.compare_segmentations, levelA=levelA, levelB=levelB,
                     labelsA=labelsA, labelsB=labelsB, cb=self.on_analysis,
                     dice=self.dice.isChecked(), jacc=self.jacc.isChecked(),
                     cohen=self.cohen.isChecked(),
                     caption='Comparing segmentations')

    def on_analysis(self, result):
        self.results.setResults(result)
