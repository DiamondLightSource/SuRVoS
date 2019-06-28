
import numpy as np
from ..qt_compat import QtGui, QtCore, QtWidgets
import os.path as op
import logging as log

from matplotlib.colors import ListedColormap

from .base import Plugin, PredWidgetBase
from ..widgets import HeaderLabel, TComboBox
from .training import Uncertainty, UncertainLabelWidget
from ..core import DataModel, LayerManager, Launcher, LabelManager
from .. import actions as ac
from ..widgets import ActionButton
from ..lib._features import find_boundaries


class Predict(PredWidgetBase):
    def __init__(self):
        super(Predict, self).__init__(parent=None)

        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict)
        self.vbox.addWidget(self.btn_predict)


    def run_prediction(self, y_data, p_data, level_params, desc_params,
                       clf_params, ref_params, out_labels, out_confidence, level):

        self.launcher.run(ac.predict_only, y_data=y_data, p_data=p_data,
                          level_params=level_params, desc_params=desc_params,
                          clf_params=clf_params, ref_params=ref_params,
                          out_labels=out_labels, out_confidence=out_confidence,
                          cb=self.on_predicted,
                          caption='Predicting labels for Level {}'.format(level))

class LoadClassifier(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(LoadClassifier, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.load = ActionButton('Load Trained Classifier')
        vbox.addWidget(self.load)
        vbox.addStretch(1)
        self.setLayout(vbox)

class ApplyFilters(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(ApplyFilters, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.apply = ActionButton('Calculate Feature Channels')
        self.apply.setEnabled(False)
        vbox.addWidget(self.apply)
        vbox.addStretch(1)
        self.setLayout(vbox)

class CalcSupervoxels(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(CalcSupervoxels, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.apply = ActionButton('Calculate Supervoxels')
        self.apply.setEnabled(False)
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
        self.load_classifier.load.clicked.connect(self.on_load_classifier)

        self.apply_filters = ApplyFilters()
        vbox.addWidget(self.apply_filters)
        self.apply_filters.apply.clicked.connect(self.on_apply_filters)

        self.calc_supervox = CalcSupervoxels()
        vbox.addWidget(self.calc_supervox)
        self.calc_supervox.apply.clicked.connect(self.on_calc_supervox)

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

    def on_load_classifier(self):
        root_dir = self.DM.wspath
        input_dir = op.join(root_dir, "classifiers")
        filter = "Classifier (*.h5)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Classifier', input_dir, filter)

        success = None
        if path is not None and len(path) > 0:
            success = self.DM.load_classifier(path)
        if success:
            # Get a dictionary with all the metadata settings
            self.meta_data_result = self.DM.load_saved_settings_from_file(path)

            ##### Labels #####
            self.load_label_info()
            level_list = self.LBLM.levels()
            label_list = [self.LBLM.labels(level) for level in level_list]
            label_names = []
            for label_sublist in label_list:
                for label in label_sublist:
                    label_names.append(label.name)

            self.launcher.info.emit("{} classifier loaded."
                                    "\nLevel {} loaded:\n{}".format(self.DM.clf_name, level_list, label_names))
            self.apply_filters.apply.setEnabled(True)

        else:
            self.launcher.error.emit("No classifier was loaded. Not applying settings")


    def load_label_info(self):
        params = self.meta_data_result['lbl_attrs']
        levelid = params['levelid']
        # First create an empty dataset
        dataset = 'annotations/annotations{}'.format(levelid)
        self.DM.create_empty_dataset(dataset, shape=self.DM.data_shape,
                                     dtype=np.float32, params=params,
                                     fillvalue=np.nan)
        self.LBLM.loadLevel(levelid, dataset)

    def on_apply_filters(self):
        self.calculate_feature_channels()

    def on_calc_supervox(self):
        self.calculate_supervoxels()

    def calculate_feature_channels(self):
        """
        
        :param result: 
        :return: 
        """
        # Check whether feature channels already exist
        if len(self.DM.available_channels()) == 0:
            features = []
            channels = self.meta_data_result['channel_list']
            dialog_result = self.confirm_channel_compute(channels)
            if dialog_result == QtWidgets.QMessageBox.Yes:

                # For each feature, create an empty file and generate a dictionary of parameters
                for channel in channels:
                    metadata = self.meta_data_result[channel]
                    initial_params = dict(active=False, feature_idx=metadata['feature_idx'],
                                          feature_type=metadata['feature_type'], feature_name=metadata['feature_name'])
                    if len(initial_params) > 0:
                        self.DM.create_empty_dataset(metadata['out'], shape=self.DM.data_shape,
                                                     dtype=np.float32, params=initial_params,
                                                     fillvalue=np.nan)
                    clamp = None
                    if metadata['clamp'] == True:
                        clamp = metadata['evmin'], metadata['evmax']
                    features.append({
                        'name': metadata['feature_name'],
                        'source': metadata['source'],
                        'clamp': clamp,
                        'params': metadata,
                        'out': metadata['out'],
                        'idx': metadata['feature_idx'],
                        'feature': metadata['feature_type']
                    })
                # Compute the feature channels
                self.launcher.run(ac.compute_all_channel, features=features,
                                  caption='Computing Multiple Features',
                                  cb=self.on_channels_calculated)
            else:
                log.info("Generation of feature channels cancelled")
        else:
            self.launcher.info.emit('Feature channels already exist. Not calculating new ones.')
            self.calc_supervox.apply.setEnabled(True)

    def confirm_channel_compute(self, channels):
        """
        
        :return: 
        """
        return QtWidgets.QMessageBox.question(self,
                                            "Confirm calculation",
                                            "The following feature channels"
                                            "\nwill be calculated:\n\n{}".format("\n".join(channels)),
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    def on_channels_calculated(self, results):
        """
        Once the channels have been calculated, add them to the Feature Channels tab
        Note: This is called multiple times - each time a channel is calculated
        :param results: 
        :return: 
        """
        for result in results:
            if result is None:
                continue
            out, idx, params = result
            self.DM.channel_computed.emit(out, params)
            fname = params['feature_name']
            ftype = params['feature_type']
            active = params['active']
            name = out.split('/')[1]
            keys = list(params.keys())
            keys.remove('feature_idx')
            keys.remove('feature_name')
            keys.remove('feature_type')
            keys.remove('active')
            channel_params = {k: params[k] for k in keys}
            log.info('* Channel {} {}'.format(fname, ftype))
            self.DM.clf_channel_computed.emit(idx, name, ftype, active, channel_params)
            self.calc_supervox.apply.setEnabled(True)

    def calculate_supervoxels(self):

        ##### Supervoxels #####

        sv_attrs = self.meta_data_result['sv_attrs']

        in_data = sv_attrs['source']
        self.sv_params = {
            'sp_shape': tuple(sv_attrs['sp_shape']),
            'spacing': tuple(sv_attrs['spacing']),
            'compactness': sv_attrs['compactness']
        }

        out_sv = 'supervoxels/supervoxels'
        out_svindex = 'supervoxels/supervoxels_idx'
        out_svtable = 'supervoxels/supervoxels_table'
        out_svedges = 'supervoxels/graph_edges'
        out_svweights = 'supervoxels/graph_edge_weights'

        self.DM.remove_dataset(out_sv)
        self.DM.remove_dataset(out_svindex)
        self.DM.remove_dataset(out_svtable)
        self.DM.remove_dataset(out_svedges)
        self.DM.remove_dataset(out_svweights)

        Launcher.instance().run(ac.create_supervoxels,
                                dataset=in_data,
                                out_sv=out_sv,
                                out_svindex=out_svindex, out_svtable=out_svtable,
                                out_svedges=out_svedges, out_svweights=out_svweights,
                                cb=self.on_supervoxels,
                                caption='Creating SuperVoxels..',
                                **self.sv_params)

    def on_supervoxels(self, params):
        svlabels, svtotal, sortindex, sorttable, edges, weights = params
        self.update_supervoxel_layer(svlabels, sortindex, sorttable, svtotal, visible=True)

    def update_supervoxel_layer(self, svlabels, sortindex, sorttable, total_sv, visible=False):
        if svlabels is not None:
            self.DM.svlabels = svlabels
            self.DM.svtotal = total_sv
            self.DM.svindex = sortindex
            self.DM.svtable = sorttable

        if self.LM.isin('SuperVoxels', 'Super-Regions'):
            self.LM.get('SuperVoxels', 'Super-Regions').data = self.DM.svlabels
            self.LM.setVisible('SuperVoxels', 'Super-Regions', True)
        else:
            cmap = ListedColormap(['MidnightBlue'])
            self.LM.addLayer(self.DM.svlabels, 'SuperVoxels', level='Super-Regions',
                             pre=find_boundaries, cmap=cmap, background=0,
                             vmin=0.1, vmax=1, visible=visible)
        self.LM.update()

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
