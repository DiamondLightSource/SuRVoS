
import numpy as np
from ..qt_compat import QtGui, QtCore, QtWidgets
import os.path as op
import logging as log

from matplotlib.colors import ListedColormap

from .base import Plugin
from ..widgets import HWidgets, PLineEdit, HeaderLabel, SubHeaderLabel, TComboBox, MultiSourceCombo, \
                      CSlider, HSize3D, BLabel, ColorButton
from ..core import DataModel, LayerManager, Launcher, LabelManager
from .. import actions as ac
from ..widgets import ActionButton

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


class Predict(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Predict, self).__init__(parent=None)

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

        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict)
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

        vbox.addWidget(HWidgets(None, self.use_region, stretch=[1, 0]))
        vbox.addWidget(HWidgets(None, 'Features:', self.use_desc, stretch=[1, 0, 0]))
        self.supervoxel_desc_params = HWidgets(self.desc_type, 'Bins:',
                                               self.desc_bins, self.nh_order,
                                               stretch=[1, 0, 0, 1])
        vbox.addWidget(self.supervoxel_desc_params)
        vbox.addWidget(HWidgets(None, self.preprocess, stretch=[1, 0, 0]))

        # vbox.addWidget(SubHeaderLabel('Classifier'))
        #
        # self.train_alg = TComboBox('Classifier type:', [
        #     'Ensemble',
        #     'SVM',
        #     'Online Linear'
        # ])
        # self.train_alg.currentIndexChanged.connect(self.on_classifier_changed)
        # vbox.addWidget(self.train_alg)


        vbox.addWidget(SubHeaderLabel('Prediction Refinement'))
        self.refinement_combo = TComboBox('Refine:', ['None', 'Potts',
                                                      'Appearance', 'Edge'])
        self.refinement_lamda = PLineEdit(50., parse=float)
        self.refinement_lamda.setMaximumWidth(80)
        self.refinement_desc = MultiSourceCombo()
        self.ref_features = HWidgets(None, 'Features:', self.refinement_desc,
                                     stretch=[1, 0, 0])
        self.ref_features.setVisible(False)
        vbox.addWidget(HWidgets(self.refinement_combo, 'Lambda:', self.refinement_lamda))
        vbox.addWidget(self.ref_features)
        vbox.addWidget(self.btn_predict)

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

    # def on_classifier_changed(self, idx):
    #     if idx == 0:
    #         self.clf_container.layout().addWidget(self.ensembles)
    #         self.svm.setParent(None)
    #         self.online.setParent(None)
    #     elif idx == 1:
    #         self.clf_container.layout().addWidget(self.svm)
    #         self.ensembles.setParent(None)
    #         self.online.setParent(None)
    #     else:
    #         self.clf_container.layout().addWidget(self.online)
    #         self.svm.setParent(None)
    #         self.ensembles.setParent(None)

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

        self.launcher.run(ac.predict_only, y_data=y_data, p_data=p_data,
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

class LoadClassifier(QtWidgets.QWidget):

    load = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(LoadClassifier, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.apply = ActionButton('Load Trained Classifier')
        vbox.addWidget(self.apply)
        vbox.addStretch(1)
        self.setLayout(vbox)


class PretrainedClassifier(Plugin):

    name = 'Pretrained classifier'

    def __init__(self, parent=None):
        super(PretrainedClassifier, self).__init__(ptype=Plugin.Plugin, parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.launcher = Launcher.instance()

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        loadLabel = HeaderLabel('Use Pre-Trained Classifier')
        vbox.addWidget(loadLabel)
        self.load_classifier =  LoadClassifier()
        vbox.addWidget(self.load_classifier)
        self.load_classifier.apply.clicked.connect(self.on_load_label)

        # Levels
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
        self.predict_widget = Predict()
        vbox.addWidget(self.predict_widget)

        vbox.addWidget(HeaderLabel('Update annotations'))
        self.uncertainty_widget = Uncertainty()
        vbox.addWidget(self.uncertainty_widget)
        vbox.addStretch(1)

    def on_load_label(self):
        print("Button Clicked!")
        root_dir = self.DM.wspath
        input_dir = op.join(root_dir, "classifiers")
        filter = "Classifier (*.joblib)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Classifier', input_dir, filter)
        if path is not None and len(path) > 0:
            self.DM.load_classifier(path)

    def on_level_selected(self, idx):
        if idx < 0:
            self.predict_widget.select_level(None, None, -1)
            return
        self.predict_widget.select_level(list(self.levels)[idx], None, -1)

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
            self.predict_widget.parent_level = self.parent_labels[idx][0]
            self.predict_widget.parent_label = self.parent_labels[idx][1].idx
        else:
            self.predict_widget.parent_level = None
            self.predict_widget.parent_label = -1

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
                if self.predict_widget.parent_level == level and \
                        self.predict_widget.parent_label == label:
                    self.use_parent.setCurrentIndex(0)
                del self.parent_labels[idx]
                self.use_parent.removeItem(idx)
