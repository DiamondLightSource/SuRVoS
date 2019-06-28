import numpy as np

import logging as log

from ..qt_compat import QtGui, QtCore, QtWidgets

from ..widgets import HWidgets, PLineEdit, TComboBox, \
    SubHeaderLabel,MultiSourceCombo
from ..core import DataModel, LayerManager, LabelManager, Launcher

class Plugin(QtWidgets.QScrollArea):

    name = "Plugin"
    header = None

    Widget = 'widget'
    Plugin = 'plugin'

    completed = QtCore.pyqtSignal(bool)

    def __init__(self, ptype=Plugin, height=0, width=300, dock='left', parent=None):

        super(Plugin, self).__init__(parent=parent)

        self.ptype = ptype
        self.container = QtWidgets.QWidget()
        self.container.setWindowTitle(self.name)
        self.setWidget(self.container)
        self.setWidgetResizable(True)

        self.layout = QtWidgets.QGridLayout(self.container)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        if ptype == self.Plugin:
            self.layout.setContentsMargins(0,10,0,0)
        self.container.setLayout(self.layout)
        self.row = 0
        self.tab_idx = 0

    def setup(self):
        self.header.setEnabled(True)
        self.header.setVisible(True)

    def addWidget(self, widget, align=QtCore.Qt.AlignTop):
        self.layout.addWidget(widget, self.row, 0, align)
        self.row += 1

    def __add__(self, widget):
        self.addWidget(widget)
        return self

    def toggle(self):
        self.setVisible(not self.isVisible())

    def value(self):
        pass

    def on_focus(self):
        pass

    def on_tab_changed(self, idx):
        pass


class PredWidgetBase(QtWidgets.QWidget):
    """
    Base class for the Predict and TrainPredict Widgets
    """

    def __init__(self, parent=None):
        super(PredWidgetBase, self).__init__(parent=None)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.launcher = Launcher.instance()
        self.selected_level = -1
        self.parent_level = None
        self.parent_label = -1

        self.vbox = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbox)

        self.vbox.addWidget(SubHeaderLabel('Descriptor'))

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

        self.vbox.addWidget(HWidgets(None, self.use_region, stretch=[1, 0]))
        self.vbox.addWidget(HWidgets(None, 'Features:', self.use_desc, stretch=[1, 0, 0]))
        self.supervoxel_desc_params = HWidgets(self.desc_type, 'Bins:',
                                               self.desc_bins, self.nh_order,
                                               stretch=[1, 0, 0, 1])
        self.vbox.addWidget(self.supervoxel_desc_params)
        self.vbox.addWidget(HWidgets(None, self.preprocess, stretch=[1, 0, 0]))

        self.vbox.addWidget(SubHeaderLabel('Prediction Refinement'))
        self.refinement_combo = TComboBox('Refine:', ['None', 'Potts',
                                                      'Appearance', 'Edge'])
        self.refinement_lamda = PLineEdit(50., parse=float)
        self.refinement_lamda.setMaximumWidth(80)
        self.refinement_desc = MultiSourceCombo()
        self.ref_features = HWidgets(None, 'Features:', self.refinement_desc,
                                     stretch=[1, 0, 0])
        self.ref_features.setVisible(False)
        self.vbox.addWidget(HWidgets(self.refinement_combo, 'Lambda:', self.refinement_lamda))
        self.vbox.addWidget(self.ref_features)

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
        self.DM.add_desc_params_to_model(desc_params)

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

        self.run_prediction(y_data, p_data, level_params, desc_params,
                            clf_params, ref_params, out_labels, out_confidence, level)

    def run_prediction(self, y_data, p_data, level_params, desc_params,
                       clf_params, ref_params, out_labels, out_confidence, level):
        """
        To be implemented in subclasses
        """
        raise NotImplementedError

    def on_predicted(self, params):
        predictions, uncertainty, labels = params
        if not self.LM.isin('Predictions', 'Predictions'):
            self.LM.addLayer(predictions, 'Predictions', 'Predictions', alpha=0.5, visible=True)
        self.DM.level_predicted.emit(self.selected_level)
