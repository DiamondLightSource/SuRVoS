import numpy as np
from ..qt_compat import QtGui, QtCore, QtWidgets

import logging as log
from collections import defaultdict

from matplotlib.colors import ListedColormap

from skimage.segmentation import find_boundaries

from ..core import Launcher
from ..widgets import LCheckBox, HWidgets, BLabel, PLineEdit, TComboBox, \
                      HeaderLabel, SubHeaderLabel, CSlider, ColorButton, \
                      HSize3D, MultiSourceCombo, ActionButton
from ..widgets.conficence_viewer import ConfidenceViewer
from .base import Plugin
from ..core import DataModel, LayerManager, LabelManager
from .. import actions as ac


class EnsembleWidget(QtWidgets.QWidget):

    predict = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(EnsembleWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
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
                                stretch=[0,1,0,1]))

        vbox.addWidget(HWidgets(QtWidgets.QLabel('Learn Rate:'),
                                self.lrate,
                                QtWidgets.QLabel('Subsample:'),
                                self.subsample,
                                stretch=[0,1,0,1]))

        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        self.n_jobs = PLineEdit(1, parse=int)
        vbox.addWidget(HWidgets('Num Jobs', self.n_jobs, None, self.btn_predict,
                                stretch=[0, 0,1,0]))

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

    def on_predict_clicked(self):
        ttype = ['rf', 'erf', 'ada', 'gbf']
        params = {
            'clf'           : 'ensemble',
            'type'          : ttype[self.type_combo.currentIndex()],
            'n_estimators'  : self.ntrees.value(),
            'max_depth'     : self.depth.value(),
            'learning_rate' : self.lrate.value(),
            'subsample'     : self.subsample.value(),
            'n_jobs'        : self.n_jobs.value()
        }
        self.predict.emit(params)


class SVMWidget(QtWidgets.QWidget):

    predict = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(SVMWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
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
                                stretch=[0,1,0,1]))

        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        vbox.addWidget(HWidgets(None, self.btn_predict, stretch=[1,0]))

    def on_predict_clicked(self):
        params = {
            'clf'       : 'svm',
            'kernel'    : self.type_combo.currentText(),
            'C'         : self.penaltyc.value(),
            'gamma'     : self.gamma.value()
        }

        self.predict.emit(params)


class OnlineWidget(QtWidgets.QWidget):

    predict = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(OnlineWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        self.setLayout(vbox)

        self.type_combo = TComboBox('Loss function:', [
                                        #'hinge',
                                        'log',
                                        'modified_huber',
                                        #'squared_hinge',
                                        #'perceptron'
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
                                stretch=[0,1,0,1]))

        self.chunks = PLineEdit(None, parse=int)

        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        vbox.addWidget(HWidgets(QtWidgets.QLabel('Chunks:'),
                                self.chunks,
                                None, self.btn_predict,
                                stretch=[0,0,1,0]))

    def on_predict_clicked(self):
        params = {
            'clf'       : 'sgd',
            'loss'      : self.type_combo.currentText(),
            'penalty'   : self.penalty.currentText(),
            'alpha'     : self.alpha.value(),
            'n_iter'    : self.n_iter.value(),
            'chunks'    : self.chunks.value()
        }

        self.predict.emit(params)


class TrainPredict(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(TrainPredict, self).__init__(parent=None)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.launcher = Launcher.instance()

        self.selected_level = -1
        self.parent_level = None
        self.parent_label = -1

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        vbox.addWidget(SubHeaderLabel('Descriptor'))

        # Region Type
        self.use_region = TComboBox('Region:', ['Voxels', 'SuperVoxels'], selected=1)
        self.use_region.currentIndexChanged.connect(self.on_region_changed)
        self.use_region.setMinimumWidth(200)

        # Descriptor type
        self.use_desc = MultiSourceCombo()

        self.desc_type = TComboBox('Type:',
                                    ['Mean', 'Quantized', 'Textons',
                                     'Covar', 'Sigma Set'])
        self.desc_bins = PLineEdit(10, parse=int)
        self.desc_bins.setMaximumWidth(60)

        self.nh_order = TComboBox('Order:', [0, 1, 2], selected=0)

        self.preprocess = TComboBox('Project:',
                                    ['None', 'standard', 'random_projection',
                                     'pca', 'random_pca'])

        vbox.addWidget(HWidgets(None, self.use_region, stretch=[1,0]))
        vbox.addWidget(HWidgets(None, 'Features:', self.use_desc, stretch=[1,0,0]))
        self.supervoxel_desc_params = HWidgets(self.desc_type, 'Bins:',
                                               self.desc_bins, self.nh_order,
                                               stretch=[1,0,0,1])
        vbox.addWidget(self.supervoxel_desc_params)
        vbox.addWidget(HWidgets(None, self.preprocess, stretch=[1,0,0]))

        vbox.addWidget(SubHeaderLabel('Classifier'))

        self.train_alg = TComboBox('Classifier type:', [
                                        'Ensemble',
                                        'SVM',
                                        'Online Linear'
                                   ])
        self.train_alg.currentIndexChanged.connect(self.on_classifier_changed)
        vbox.addWidget(self.train_alg)

        self.clf_container = QtWidgets.QWidget()
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.setContentsMargins(0,0,0,0)
        self.clf_container.setLayout(vbox2)

        self.ensembles = EnsembleWidget()
        self.ensembles.predict.connect(self.on_predict)
        self.svm = SVMWidget()
        self.svm.predict.connect(self.on_predict)
        self.online = OnlineWidget()
        self.online.predict.connect(self.on_predict)

        self.clf_container.layout().addWidget(self.ensembles)
        vbox.addWidget(self.clf_container)

        vbox.addWidget(SubHeaderLabel('Prediction Refinement'))
        self.refinement_combo = TComboBox('Refine:', ['None', 'Potts',
                                                      'Appearance', 'Edge'])
        self.refinement_lamda = PLineEdit(50., parse=float)
        self.refinement_lamda.setMaximumWidth(80)
        self.refinement_desc = MultiSourceCombo()
        self.ref_features = HWidgets(None, 'Features:', self.refinement_desc,
                                     stretch=[1,0,0])
        self.ref_features.setVisible(False)
        vbox.addWidget(HWidgets(self.refinement_combo, 'Lambda:', self.refinement_lamda))
        vbox.addWidget(self.ref_features)

        self.refinement_combo.currentIndexChanged.connect(self.on_ref_type_changed)
        self.use_region.currentIndexChanged.connect(self.on_region_changed)
        self.DM.voxel_descriptor_computed.connect(self.on_voxel_desc_changed)
        self.DM.voxel_descriptor_removed.connect(self.on_voxel_desc_changed)
        self.DM.supervoxel_descriptor_computed.connect(self.on_voxel_desc_changed)
        self.DM.supervoxel_descriptor_computed.connect(self.on_voxel_desc_changed)
        self.DM.supervoxel_descriptor_cleared.connect(self.on_voxel_desc_changed)

    def select_level(self, level, plevel, plabel):
        self.selected_level = level
        self.parent_level = plevel
        self.parent_label = plabel

    def on_voxel_desc_changed(self):
        self.on_region_changed(self.use_region.currentIndex())

    def on_ref_type_changed(self, idx):
        self.ref_features.setVisible(idx > 2)

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

    def on_region_changed(self, idx):
        self.supervoxel_desc_params.setVisible(idx == 1)

    def on_predict(self, params):
        ### LEVEL AND LABEL
        level = self.selected_level

        if level < 0 or level is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No level selected")
            return

        parent_level = self.parent_level
        parent_label = self.parent_label
        idxs = self.LBLM.idxs(level=level)

        if len(idxs) == 0:
            self.launcher.error.emit('Current level doesn\'t contain any label')
            return

        level_params = dict(level=level, plevel=parent_level, plabel=parent_label)

        ### DESCRIPTOR
        desc_params = dict()
        desc_params['supervoxels'] = 'supervoxels/supervoxels' if self.use_region.currentIndex() == 1 else None
        desc_params['features'] = self.use_desc.value()
        desc_params['projection'] = self.preprocess.value()
        if desc_params['supervoxels']:
            desc_params['desc_type'] = self.desc_type.value()
            desc_params['desc_bins'] = self.desc_bins.value()
            desc_params['nh_order'] = self.nh_order.value()
            desc_params['sp_edges'] = 'supervoxels/graph_edges'

        if len(desc_params['features']) == 0:
            self.launcher.error.emit('No descriptor feature has been selected.')
            return

        ### CLASSIFIER
        clf_params = params

        ### REFINEMENT
        ref_params = dict()
        ref_params['ref_type'] = self.refinement_combo.value()
        ref_params['lambda'] = self.refinement_lamda.value()
        ref_params['features'] = self.refinement_desc.value()
        ref_params['sp_edges'] = 'supervoxels/graph_edges'
        ref_params['sp_eweights'] = 'supervoxels/graph_edge_weights'

        if self.refinement_combo.currentIndex() > 2 and len(ref_params['features']) == 0:
            self.launcher.error.emit('No refinement feature has been selected.')
            return

        ### ANNOTATIONS
        y_data = self.LBLM.dataset(level)
        out_labels = 'predictions/predictions'
        out_confidence = 'predictions/confidence'

        if parent_level != None and parent_level >= 0 and parent_label >= 0:
            p_data = self.LBLM.dataset(parent_level)
        else:
            p_data = None

        self.DM.remove_dataset(out_labels)
        self.DM.remove_dataset(out_confidence)

        dataset = 'predictions/predictions'
        self.DM.create_empty_dataset(out_labels, self.DM.data_shape,
                                     np.int16, check=False)
        self.DM.create_empty_dataset(out_confidence, self.DM.data_shape,
                                     np.float32, check=False)

        log.info("Level {}".format(level_params))
        log.info("Desc {}".format(desc_params))
        log.info("Clf {}".format(clf_params))
        log.info("Ref {}".format(ref_params))
        log.info("{} {}".format(y_data, p_data))
        log.info("{} {}".format(out_labels, out_confidence))

        self.launcher.run(ac.predict_proba, y_data=y_data, p_data=p_data,
                          level_params=level_params, desc_params=desc_params,
                          clf_params=clf_params, ref_params=ref_params,
                          out_labels=out_labels, out_confidence=out_confidence,
                          cb=self.on_predicted,
                          caption='Predicting labels for Level {}'.format(level))

    def on_predicted(self, params):
        predictions, uncertainty, labels = params
        if not self.LM.isin('Predictions', 'Predictions'):
            self.LM.addLayer(predictions, 'Predictions', 'Predictions', alpha=0.5, visible=True)
        self.DM.level_predicted.emit(self.selected_level)


class UncertainLabelWidget(QtWidgets.QWidget):

    save = QtCore.pyqtSignal(int)

    def __init__(self, idx, name, color, parent=None):
        super(UncertainLabelWidget, self).__init__(parent=parent)

        self.idx = idx

        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(hbox)

        self.name = BLabel(name)
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
        vbox.addWidget(HWidgets('Confidence:', self.slider_thresh, stretch=[0,1]))

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
            maxid = 0 if len(idxs) == 0 else max(idxs)+1
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

    name = 'Model training'

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

        self.addWidget(dummy)
        self.addWidget(HeaderLabel('Predict labels'))
        self.train_widget = TrainPredict()
        self.addWidget(self.train_widget)

        self.addWidget(HeaderLabel('Update annotations'))
        self.uncertainty_widget = Uncertainty()
        self.addWidget(self.uncertainty_widget)

        self.parent_labels = []
        self.levels = self.LBLM.levels()

    def on_level_selected(self, idx):
        if idx < 0:
            self.train_widget.select_level(None, None, -1)
            return
        self.train_widget.select_level(self.levels[idx], None, -1)

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
        if current >= 0 and self.levels[current] != level:
            self.use_level.setCurrentIndex(self.levels.index(level))

    def on_label_name_changed(self, level, ds, label, name):
        current = self.use_level.currentIndex()
        if current >= 0 and self.levels[current] != level:
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
        if current >= 0 and self.levels[current] != level:
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
