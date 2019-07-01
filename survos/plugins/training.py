import numpy as np
import os
import os.path as op
from ..qt_compat import QtGui, QtCore, QtWidgets

import logging as log

from matplotlib.colors import ListedColormap

from ..core import Launcher
from ..widgets import LCheckBox, HWidgets, BLabel, PLineEdit, TComboBox, \
    HeaderLabel, SubHeaderLabel, CSlider, ColorButton, \
    HSize3D, ActionButton
from .base import Plugin, PredWidgetBase
from ..core import DataModel, LayerManager, LabelManager
from .. import actions as ac


class EnsembleWidget(QtWidgets.QWidget):
    train_predict = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(EnsembleWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.type_combo = TComboBox('Ensemble Type:', [
            'Random Forest',
            'ExtraRandom Forest',
            'AdaBoost',
            'GradientBoosting'
        ])
        self.type_combo.currentIndexChanged.connect(self.on_ensemble_changed)
        vbox.addWidget(self.type_combo)

        self.ntrees = PLineEdit(100, parse=int)
        self.depth = PLineEdit(None, parse=int)
        self.lrate = PLineEdit(1., parse=float)
        self.subsample = PLineEdit(1., parse=float)

        vbox.addWidget(HWidgets(QtWidgets.QLabel('# Trees:'),
                                self.ntrees,
                                QtWidgets.QLabel('Max Depth:'),
                                self.depth,
                                stretch=[0, 1, 0, 1]))

        vbox.addWidget(HWidgets(QtWidgets.QLabel('Learn Rate:'),
                                self.lrate,
                                QtWidgets.QLabel('Subsample:'),
                                self.subsample,
                                stretch=[0, 1, 0, 1]))

        self.btn_train_predict = ActionButton('Train && Predict')
        # self.btn_predict = ActionButton('Predict')
        # self.btn_predict.setEnabled(False)
        self.btn_train_predict.clicked.connect(self.on_train_predict_clicked)
        # self.btn_predict.clicked.connect(self.on_predict_clicked)
        self.n_jobs = PLineEdit(1, parse=int)
        vbox.addWidget(HWidgets('Num Jobs', self.n_jobs, None, self.btn_train_predict,
                                stretch=[0, 0, 1, 0]))
        # vbox.addWidget(HWidgets(None, None, None, self.btn_predict,
        #                         stretch=[0, 0, 1, 0]))

    def on_ensemble_changed(self, idx):
        if idx == 2:
            self.ntrees.setDefault(50)
        else:
            self.ntrees.setDefault(100)

        if idx == 3:
            self.lrate.setDefault(0.1)
            self.depth.setDefault(3)
        else:
            self.lrate.setDefault(1.)
            self.depth.setDefault(None)

    def on_train_predict_clicked(self):
        ttype = ['rf', 'erf', 'ada', 'gbf']
        params = {
            'clf': 'ensemble',
            'type': ttype[self.type_combo.currentIndex()],
            'n_estimators': self.ntrees.value(),
            'max_depth': self.depth.value(),
            'learning_rate': self.lrate.value(),
            'subsample': self.subsample.value(),
            'n_jobs': self.n_jobs.value()
        }
        self.train_predict.emit(params)


class SVMWidget(QtWidgets.QWidget):
    predict = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(SVMWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.type_combo = TComboBox('Kernel Type:', [
            'linear',
            'poly',
            'rbf',
            'sigmoid'
        ])
        vbox.addWidget(self.type_combo)

        self.penaltyc = PLineEdit(1.0, parse=float)
        self.gamma = PLineEdit('auto', parse=float)

        vbox.addWidget(HWidgets(QtWidgets.QLabel('Penalty C:'),
                                self.penaltyc,
                                QtWidgets.QLabel('Gamma:'),
                                self.gamma,
                                stretch=[0, 1, 0, 1]))

        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        vbox.addWidget(HWidgets(None, self.btn_predict, stretch=[1, 0]))

    def on_predict_clicked(self):
        params = {
            'clf': 'svm',
            'kernel': self.type_combo.currentText(),
            'C': self.penaltyc.value(),
            'gamma': self.gamma.value()
        }

        self.predict.emit(params)


class OnlineWidget(QtWidgets.QWidget):
    predict = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(OnlineWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.type_combo = TComboBox('Loss function:', [
            # 'hinge',
            'log',
            'modified_huber',
            # 'squared_hinge',
            # 'perceptron'
        ])
        vbox.addWidget(self.type_combo)

        self.penalty = TComboBox('Regularization:', [
            'none',
            'l1',
            'l2',
            'elasticnet',
        ])
        vbox.addWidget(self.penalty)

        self.alpha = PLineEdit(0.0001, parse=float)
        self.n_iter = PLineEdit(5, parse=int)

        vbox.addWidget(HWidgets(QtWidgets.QLabel('alpha:'),
                                self.alpha,
                                QtWidgets.QLabel('Num Iter:'),
                                self.n_iter,
                                stretch=[0, 1, 0, 1]))

        self.chunks = PLineEdit(None, parse=int)

        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        vbox.addWidget(HWidgets(QtWidgets.QLabel('Chunks:'),
                                self.chunks,
                                None, self.btn_predict,
                                stretch=[0, 0, 1, 0]))

    def on_predict_clicked(self):
        params = {
            'clf': 'sgd',
            'loss': self.type_combo.currentText(),
            'penalty': self.penalty.currentText(),
            'alpha': self.alpha.value(),
            'n_iter': self.n_iter.value(),
            'chunks': self.chunks.value()
        }

        self.predict.emit(params)


class SaveClassifier(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SaveClassifier, self).__init__(parent=parent)

        self.DM = DataModel.instance()
        self.LBLM = LabelManager.instance()
        self.selected_level = -1

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        self.btn_save_clf = ActionButton('Save Classifier to Disk')
        self.btn_save_clf.clicked.connect(self.on_save_clf_clicked)
        vbox.addWidget(HWidgets(None, None, self.btn_save_clf, None,
                                stretch=[0, 0, 1, 0]))
        self.DM.level_predicted.connect(self.on_level_predicted)

    def on_level_predicted(self, idx):
        self.select_level(idx)

    def select_level(self, idx):
        self.selected_level = idx

    def on_save_clf_clicked(self):
        if not self.DM.has_classifier():
            QtWidgets.QMessageBox.critical(self, "Error", "A classifier has not been created yet!")
            return
        else:
            attrs = {'levelid': self.selected_level, 'label': self.LBLM.idxs(self.selected_level),
                     'names': self.LBLM.names(self.selected_level), 'colors': self.LBLM.colors(self.selected_level),
                     'visible': list(map(int, self.LBLM.visibility(self.selected_level))),
                     'parent_levels': self.LBLM.parent_levels(self.selected_level),
                     'parent_labels': self.LBLM.parent_labels(self.selected_level)}

            root_dir = self.DM.wspath
            output_dir = op.join(root_dir, "classifiers")
            os.makedirs(output_dir, exist_ok=True)
            filename = op.join(output_dir, "classifier.h5")
            filter = "Classifier (*.h5)"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Classifier', filename, filter)
            if path is not None and len(path) > 0:
                log.info('+ Saving classifier to {}'.format(path))
                self.DM.save_classifier(path, attrs)


class TrainPredict(PredWidgetBase):
    def __init__(self):
        super(TrainPredict, self).__init__(parent=None)

        self.vbox.addWidget(SubHeaderLabel('Classifier'))

        self.train_alg = TComboBox('Classifier type:', [
            'Ensemble',
            'SVM',
            'Online Linear'
        ])
        self.train_alg.currentIndexChanged.connect(self.on_classifier_changed)
        self.vbox.addWidget(self.train_alg)

        self.clf_container = QtWidgets.QWidget()
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.setContentsMargins(0, 0, 0, 0)
        self.clf_container.setLayout(vbox2)

        self.ensembles = EnsembleWidget()
        self.ensembles.train_predict.connect(self.on_predict)
        self.svm = SVMWidget()
        self.svm.predict.connect(self.on_predict)
        self.online = OnlineWidget()
        self.online.predict.connect(self.on_predict)

        self.clf_container.layout().addWidget(self.ensembles)
        self.vbox.addWidget(self.clf_container)

    def on_classifier_changed(self, idx):
        if idx == 0:
            self.clf_container.layout().addWidget(self.ensembles)
            self.svm.setParent(None)
            self.online.setParent(None)
        elif idx == 1:
            self.clf_container.layout().addWidget(self.svm)
            self.ensembles.setParent(None)
            self.online.setParent(None)
        else:
            self.clf_container.layout().addWidget(self.online)
            self.svm.setParent(None)
            self.ensembles.setParent(None)

    def run_prediction(self, y_data, p_data, level_params, desc_params,
                       clf_params, ref_params, out_labels, out_confidence, level):

        self.launcher.run(ac.predict_proba, y_data=y_data, p_data=p_data,
                          level_params=level_params, desc_params=desc_params,
                          clf_params=clf_params, ref_params=ref_params,
                          out_labels=out_labels, out_confidence=out_confidence,
                          cb=self.on_predicted,
                          caption='Predicting labels for Level {}'.format(level))


class UncertainLabelWidget(QtWidgets.QWidget):
    save = QtCore.pyqtSignal(int)

    def __init__(self, idx, name, color, parent=None):
        super(UncertainLabelWidget, self).__init__(parent=parent)

        self.idx = idx

        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(hbox)

        self.name = BLabel(str(name))
        self.name.setMinimumWidth(120)
        hbox.addWidget(self.name)

        self.color = ColorButton(color, clickable=False)
        hbox.addWidget(self.color, 1)

        self.apply = ActionButton('Save')
        hbox.addWidget(self.apply)

        self.apply.clicked.connect(self.on_save_label)

    def on_save_label(self):
        self.save.emit(self.idx)


class Uncertainty(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Uncertainty, self).__init__(parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()

        self.selected_level = -1
        self.parent_level = None
        self.parent_label = -1

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        self.level_header = SubHeaderLabel('No Level Predicted Yet')
        vbox.addWidget(self.level_header)

        self.combo_viz = TComboBox('Visualization:', ['Predictions', 'Confidence'])
        vbox.addWidget(self.combo_viz)

        self.labels = {}

        self.slider_thresh = CSlider('none', current=0)
        vbox.addWidget(HWidgets('Confidence:', self.slider_thresh, stretch=[0, 1]))

        self.txt_from = HSize3D('From', default=(0, 0, 0), txtwidth=50)
        self.txt_to = HSize3D('To', default=self.DM.data_shape, txtwidth=50)
        vbox.addWidget(self.txt_from)
        vbox.addWidget(self.txt_to)

        self.combo_viz.currentIndexChanged.connect(self.on_viz_changed)
        self.slider_thresh.valueChanged.connect(self.on_thresh_changed)
        self.DM.level_predicted.connect(self.on_level_predicted)

    def on_level_predicted(self, idx):
        self.select_level(idx)

    def on_viz_changed(self, idx):
        if not self.LM.isin('Predictions', 'Predictions'):
            return

        labels_name = 'predictions/predictions'
        confidence_name = 'predictions/confidence'
        if idx == 0 and self.DM.attr(labels_name, 'active') == True:
            data = labels_name
            labels = self.DM.attr(labels_name, 'labels')
            colors = self.LBLM.colors(self.selected_level)
            idxs = self.LBLM.idxs(self.selected_level)
            maxid = 0 if len(idxs) == 0 else max(idxs) + 1
            cmaplist = ['#000000'] * maxid
            for idx, color in zip(idxs, colors):
                cmaplist[idx] = color
            cmap = ListedColormap(cmaplist)
        else:
            data = confidence_name
            cmap = 'viridis'
            maxid = 1

        layer = self.LM.get('Predictions', 'Predictions')
        layer.data = data
        layer.cmap = cmap
        layer.vmin = 0
        layer.vmax = maxid

        if self.combo_viz.currentIndex() == 1:
            layer.threshold = None
        else:
            layer.threshold = (confidence_name, self.slider_thresh.value() / 100.)

        self.LM.setVisible('Predictions', 'Predictions', True)
        self.LM.update()

    def on_thresh_changed(self, thresh):
        self.on_viz_changed(self.combo_viz.currentIndex())

    def select_level(self, level):
        self.level_header.setText('Confidence Tool for Level {}'.format(level))
        self.selected_level = level

        for labelobj in self.labels.values():
            labelobj.setParent(None)

        self.labels = {}

        if level >= 0:
            for label in self.LBLM.labels(level):
                labelobj = UncertainLabelWidget(label.idx, label.name, label.color)
                labelobj.save.connect(self.on_save_label)
                self.layout().addWidget(labelobj)
                self.labels[label.idx] = labelobj

        self.on_viz_changed(self.combo_viz.currentIndex())

    def on_save_label(self, idx):
        if not self.LM.isin('Predictions', 'Predictions'):
            QtWidgets.QMessageBox.critical(self, 'Error', 'No label predicted')
            return

        f = self.txt_from.value()
        t = self.txt_to.value()
        s1 = slice(f[0], t[0])
        s2 = slice(f[1], t[1])
        s3 = slice(f[2], t[2])

        labels_name = 'predictions/predictions'
        uncertainty_name = 'predictions/confidence'
        threshold = self.slider_thresh.value() / 100.

        launcher = Launcher.instance()
        launcher.setup('Saving predictions for [{}]'.format(self.selected_level))
        log.info('+ Loading data into memory')
        pred = self.DM.load_slices(labels_name)
        conf = self.DM.load_slices(uncertainty_name)

        log.info('+ Masking annotations')
        mask = np.zeros(pred.shape, dtype=np.bool)
        np.logical_and((pred[s1, s2, s3] == idx), (conf[s1, s2, s3] >= threshold),
                       out=mask[s1, s2, s3])

        log.info('+ Saving results to disk')
        dataset = self.LBLM.dataset(self.selected_level)
        target = self.DM.load_slices(dataset)
        target[mask] = idx
        self.DM.write_slices(dataset, target)
        self.LM.update()

        launcher.cleanup()


class Training(Plugin):
    name = 'Train classifier'

    def __init__(self, parent=None):
        super(Training, self).__init__(ptype=Plugin.Plugin, parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.launcher = Launcher.instance()

        # Levels
        dummy = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        dummy.setLayout(vbox)
        self.use_level = TComboBox('Predict Level:', ['Level {}'.format(i) for i in self.LBLM.levels()])
        self.use_parent = TComboBox('Constrain Region:', ['None'])

        self.use_level.currentIndexChanged.connect(self.on_level_selected)
        self.use_parent.currentIndexChanged.connect(self.on_parent_selected)

        vbox.addWidget(self.use_level)
        vbox.addWidget(self.use_parent)

        self.LBLM.levelAdded.connect(self.on_levels_changed)
        self.LBLM.levelLoaded.connect(self.on_levels_changed)
        self.LBLM.levelRemoved.connect(self.on_levels_changed)

        self.LBLM.labelAdded.connect(self.on_label_added)
        self.LBLM.labelLoaded.connect(self.on_label_added)
        self.LBLM.labelNameChanged.connect(self.on_label_name_changed)
        self.LBLM.labelRemoved.connect(self.on_label_removed)

        self.DM.level_predicted.connect(self.on_level_predicted)

        self.addWidget(dummy)
        self.addWidget(HeaderLabel('Train New Classifier'))
        self.train_widget = TrainPredict()
        self.addWidget(self.train_widget)

        self.addWidget(HeaderLabel('Update annotations'))
        self.uncertainty_widget = Uncertainty()
        self.addWidget(self.uncertainty_widget)

        self.parent_labels = []
        self.levels = self.LBLM.levels()

        self.addWidget(HeaderLabel('Save Trained Classifier'))
        self.save_classifier_widget = SaveClassifier()
        self.addWidget(self.save_classifier_widget)
        if not self.DM.has_classifier():
            self.save_classifier_widget.btn_save_clf.setEnabled(False)

    def on_level_selected(self, idx):
        if idx < 0:
            self.train_widget.select_level(None, None, -1)
            return
        self.train_widget.select_level(list(self.levels)[idx], None, -1)

        self.parent_labels = [(None, None)]
        self.use_parent.clear()
        self.use_parent.addItem('None')
        for i in range(len(self.levels)):
            if i == idx:
                continue
            prev_lvl = self.use_level.itemText(i)
            for label in self.LBLM.labels(self.levels[i]):
                self.parent_labels += [(self.levels[i], label)]
                self.use_parent.addItem('{}/{}'.format(prev_lvl, label.name))

    def on_parent_selected(self, idx):
        if idx > 0:
            self.train_widget.parent_level = self.parent_labels[idx][0]
            self.train_widget.parent_label = self.parent_labels[idx][1].idx
        else:
            self.train_widget.parent_level = None
            self.train_widget.parent_label = -1

    def on_levels_changed(self):
        self.levels = self.LBLM.levels()
        self.use_level.updateItems(['Level {}'.format(i) for i in self.levels])
        self.parent_level = None

    def on_label_added(self, level, dataset, label):
        current = self.use_level.currentIndex()
        if current >= 0 and list(self.levels)[current] != level:
            self.use_level.setCurrentIndex(self.levels.index(level))

    def on_label_name_changed(self, level, ds, label, name):
        current = self.use_level.currentIndex()
        if current >= 0 and list(self.levels)[current] != level:
            idx = -1
            for n, (lvl, lbl) in enumerate(self.parent_labels):
                if lvl == level and lbl.idx == label:
                    idx = n
                    break
            if idx > 0:
                prev_lvl = self.use_level.itemText(self.levels.index(lvl))
                self.use_parent.setItemText(idx, '{}/{}'.format(prev_lvl, name))

    def on_label_removed(self, level, ds, label):
        current = self.use_level.currentIndex()
        if current >= 0 and list(self.levels)[current] != level:
            idx = -1
            for n, (lvl, lbl) in enumerate(self.parent_labels):
                if lvl == level and lbl.idx == label:
                    idx = n
                    break
            if idx > 0:
                if self.train_widget.parent_level == level and \
                                self.train_widget.parent_label == label:
                    self.use_parent.setCurrentIndex(0)
                del self.parent_labels[idx]
                self.use_parent.removeItem(idx)

    def on_level_predicted(self, level):
        self.save_classifier_widget.btn_save_clf.setEnabled(True)