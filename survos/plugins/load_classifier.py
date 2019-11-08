
import numpy as np
from ..qt_compat import QtGui, QtCore, QtWidgets
import os.path as op
import os
import logging as log
from ..core import Singleton

from matplotlib.colors import ListedColormap

from .base import Plugin, PredWidgetBase
from ..widgets import HeaderLabel, TComboBox
from .training import Uncertainty, UncertainLabelWidget
from ..core import DataModel, LayerManager, Launcher, LabelManager
from .. import actions as ac
from ..widgets import ActionButton
from ..lib._features import find_boundaries




class PredictButtonWidget(QtWidgets.QWidget):
    predict = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(PredictButtonWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)
        self.DM = DataModel.instance()
        self.btn_predict = ActionButton('Predict')
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        vbox.addWidget(self.btn_predict)

    def on_predict_clicked(self):
        # Get classifier information from file via DM
        params = self.DM.get_classifier_params()
        self.predict.emit(params)


class Predict(PredWidgetBase):
    save_signal = QtCore.pyqtSignal()

    def __init__(self):
        super(Predict, self).__init__(parent=None)

        self.predict_btn_widget = PredictButtonWidget()
        self.predict_btn_widget.predict.connect(self.on_predict)
        self.vbox.addWidget(self.predict_btn_widget)

        # Stores whether launcher post() method is connected to classifier save function
        self.sv_clf_conn = False


    def run_prediction(self, y_data, p_data, level_params, desc_params, ref_params,
                       clf_params, out_labels, out_confidence, level):
        """
        Overrides method from parent class
        """
        append = False
        train = False

        # Test for new annotations
        if self.DM.new_annotations_added(y_data):
            # 1. Ask if they would like to append new data to existing and train a new classifier
            answer = QtWidgets.QMessageBox.question(self,
                                           "Append New Data?",
                                           "Additional annotation data has been added, would you"
                                           "\nlike to train a new classifier with this data included?"
                                           "\n\nIf you chose 'No', the prediction will be repeated without\nthis new data.",
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

            if answer == QtWidgets.QMessageBox.Yes:
                # Add a signal here to enable "Save new classifier" button
                self.launcher.post.connect(self.enable_save_classifer_btn)
                self.sv_clf_conn = True
                # Append and train and predict with new classifier
                append = True
                train = True

        self.launcher.run(ac.predict_proba, y_data=y_data, p_data=p_data, append=append,
                          train=train, level_params=level_params, desc_params=desc_params,
                          clf_params=clf_params, ref_params=ref_params,
                          out_labels=out_labels, out_confidence=out_confidence,
                          cb=self.on_predicted,
                          caption='Predicting labels for Level {}'.format(level))

    def enable_save_classifer_btn(self):

        self.save_signal.emit()

        if self.sv_clf_conn:
            self.launcher.post.disconnect(self.enable_save_classifer_btn)
            self.sv_clf_conn = False


class LoadClassifier(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(LoadClassifier, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.load = ActionButton('Load Trained Classifier')
        vbox.addWidget(self.load)
        vbox.addStretch(1)
        self.setLayout(vbox)

class ClassifierInfo(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(ClassifierInfo, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.info_label = QtWidgets.QLabel()
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        my_font = QtGui.QFont()
        my_font.setBold(True)
        self.info_label.setFont(my_font)
        self.info_label.hide()
        vbox.addWidget(self.info_label)
        vbox.addStretch(1)
        self.setLayout(vbox)

    def set_text(self, text):
        self.info_label.setText(text)
        self.info_label.show()

class ApplySettings(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(ApplySettings, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.apply = ActionButton('Calculate Channels && Supervoxels')
        self.apply.setEnabled(False)
        vbox.addWidget(self.apply)
        vbox.addStretch(1)
        self.setLayout(vbox)


class SaveClassifier(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(SaveClassifier, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()

        self.save = ActionButton('Save New Classifier')
        self.save.setEnabled(False)
        vbox.addWidget(self.save)
        vbox.addStretch(1)
        self.setLayout(vbox)


@Singleton
class PretrainedClassifier(Plugin):

    name = 'Pretrained classifier'

    def __init__(self, parent=None):
        super().__init__(ptype=Plugin.Plugin, parent=parent)

        self.DM = DataModel.instance()
        self.LM = LayerManager.instance()
        self.LBLM = LabelManager.instance()
        self.launcher = Launcher.instance()

        # Stores whether launcher post() method is connected to supervoxel calculation function
        self.svx_conn = False

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        loadLabel = HeaderLabel('Use Pre-Trained Classifier')
        vbox.addWidget(loadLabel)
        self.load_classifier =  LoadClassifier()
        vbox.addWidget(self.load_classifier)
        self.load_classifier.load.clicked.connect(self.on_load_classifier)

        self.classifier_info = ClassifierInfo()
        vbox.addWidget(self.classifier_info)

        self.apply_settings = ApplySettings()
        vbox.addWidget(self.apply_settings)
        self.apply_settings.apply.clicked.connect(self.on_apply_settings)

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
        if not self.DM.has_classifier():
            self.predict_widget.predict_btn_widget.btn_predict.setEnabled(False)
        self.predict_widget.save_signal.connect(self.enable_save_button)

        vbox.addWidget(HeaderLabel('Update annotations'))
        self.uncertainty_widget = Uncertainty()
        vbox.addWidget(self.uncertainty_widget)
        vbox.addStretch(1)

        vbox.addWidget(HeaderLabel('Save Modified Classifier'))
        self.save_clf_widget = SaveClassifier()
        self.save_clf_widget.save.clicked.connect(self.on_save_clf_clicked)
        vbox.addWidget(self.save_clf_widget)
        vbox.addStretch(1)

    def enable_save_button(self):
        self.save_clf_widget.save.setEnabled(True)

    def on_save_clf_clicked(self):
        if not self.DM.has_classifier():
            QtWidgets.QMessageBox.critical(self, "Error", "A classifier has not been created yet!")
            return
        else:
            current_level = self.predict_widget.selected_level
            attrs = dict(levelid=current_level,
                         label=self.LBLM.idxs(current_level),
                         names=self.LBLM.names(current_level),
                         colors=self.LBLM.colors(current_level),
                         visible=list(map(int, self.LBLM.visibility(current_level))),
                         parent_levels=self.LBLM.parent_levels(current_level),
                         parent_labels=self.LBLM.parent_labels(current_level),
                         parent_colours=[self.LBLM.get(level, label).color for level, label in
                                         zip(self.LBLM.parent_levels(current_level),
                                             self.LBLM.parent_labels(current_level))])

            root_dir = self.DM.wspath
            output_dir = op.join(root_dir, "classifiers")
            os.makedirs(output_dir, exist_ok=True)
            filename = op.join(output_dir, "modified_classifier.h5")
            filter = "Classifier (*.h5)"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Classifier', filename, filter)
            if path is not None and len(path) > 0:
                log.info('+ Saving classifier to {}'.format(path))
                self.DM.save_classifier(path, attrs)
                # Wipe the latest set of annotations to allow for more to be added
                self.LBLM.removeLevel(current_level, force=True)
                self.load_label_info()

    def on_load_classifier(self):
        root_dir = self.DM.wspath
        input_dir = op.join(root_dir, "classifiers")
        filter = "Classifier (*.h5)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Classifier', input_dir, filter)

        success = None
        if path is not None and len(path) > 0:
            success = self.DM.load_classifier(path)
        if success:
            self.disable_train_tab()
            self.classifier_info.set_text(self.DM.clf_saved_name)
            # Get a dictionary with all the metadata settings
            self.meta_data_result = self.DM.load_saved_settings_from_file(path, has_classifier=True)

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
            self.apply_settings.apply.setEnabled(True)

        else:
            self.launcher.error.emit("No classifier was loaded. Not applying settings")

    def disable_train_tab(self):
        plugins = self.parent().findChildren(Plugin)
        for plugin in plugins:
            if plugin.name == "Train classifier":
                plugin.setEnabled(False)

    def load_label_info(self):
        params = self.meta_data_result['lbl_attrs']
        levelid = params['levelid']
        # First create an empty dataset
        dataset = 'annotations/annotations{}'.format(levelid)
        self.DM.create_empty_dataset(dataset, shape=self.DM.data_shape,
                                     dtype=np.int16, params=params,
                                     fillvalue=-1)
        self.LBLM.loadLevel(levelid, dataset)

    def on_apply_settings(self):
        # Will calculate supervoxels after feature channels calculated
        self.calculate_feature_channels()

    def calculate_feature_channels(self):
        """
        Calculates feature channels with settings retrieved from HDF5 file.
        Attaches signal to the launcher to trigger supervoxel calculation when done.
        """
        # Check whether feature channels already exist

        if len(self.DM.available_channels()) == 0:
            features = []
            channels = self.meta_data_result['channel_list']

            dialog_result = self.confirm_settings_compute()
            if dialog_result == QtWidgets.QMessageBox.Yes:
                append_later = []
                # For each feature, create an empty file and generate a dictionary of parameters
                for channel in channels:
                    if channel != 'data':
                        metadata = self.meta_data_result[channel]
                        log.info("+ Queueing {}".format(metadata['feature_name']))
                        initial_params = dict(active=False, feature_idx=metadata['feature_idx'],
                                              feature_type=metadata['feature_type'], feature_name=metadata['feature_name'])
                        if len(initial_params) > 0:
                            self.DM.create_empty_dataset(metadata['out'], shape=self.DM.data_shape,
                                                         dtype=np.float32, params=initial_params,
                                                         fillvalue=np.nan)
                        clamp = None
                        if metadata['clamp'] == True:
                            clamp = metadata['evmin'], metadata['evmax']

                        info_dict = {
                                'name': metadata['feature_name'],
                                'source': metadata['source'],
                                'clamp': clamp,
                                'params': metadata,
                                'out': metadata['out'],
                                'idx': metadata['feature_idx'],
                                'feature': metadata['feature_type']
                            }

                        if metadata['source'] != '/data': # Non standard source for the feature
                            # Search for source channel
                            found_dicts = [item for item in features if item['out'] == metadata['source']]
                            if found_dicts:
                                # It is found, carry on as before
                                features.append(info_dict)
                            else:
                                # The source isn't there, so put this feature to the back of the queue
                                append_later.append(info_dict)
                        else: # Standard source for the feature
                            features.append(info_dict)

                # Append the final dictionaries with non-standard data sources
                features += append_later

                # Connect a signal from the launcher to initiate the supervoxel calculation when done
                if self.meta_data_result.get('sv_attrs', None):
                    self.launcher.post.connect(self.calculate_supervoxels)
                    self.svx_conn = True
                # Compute the feature channels
                self.launcher.run(ac.compute_all_channel, features=features,
                                  caption='Computing Multiple Features',
                                  cb=self.on_channels_calculated)
            else:
                log.info("Generation of feature channels cancelled")
        else:
            self.launcher.info.emit('Feature channels already exist. Not calculating new ones.')
            if not self.DM.has_grp('supervoxels') and self.meta_data_result.get('sv_attrs', None):
                self.calculate_supervoxels()
            elif self.meta_data_result.get('sv_attrs', None):
                pass
            else:
                self.launcher.info.emit('Supervoxels already exist. Not calculating new ones.')
            if self.DM.has_classifier():
                self.predict_widget.predict_btn_widget.btn_predict.setEnabled(True)

    def confirm_settings_compute(self):
        """
        Helper function to display dialog box
        """
        channels = self.meta_data_result['channel_list']
        full_names = [self.meta_data_result[name]['feature_name']
                      for name in channels
                      if self.meta_data_result[name]['feature_name'] != "data"]
        supervox_params = self.meta_data_result.get('sv_attrs', None)
        supervox_string = "" if supervox_params is None else "\n\nThe following supervoxels will be" \
                                                             " calculated:\n\nShape: {}\nSpacing: {}\nCompactness:" \
                                                             " {}\nSource: {}".format(", ".join(map(str, supervox_params['sp_shape'])),
                                                             ", ".join(map(str, supervox_params['spacing'])),
                                                             supervox_params['compactness'],
                                                             supervox_params['source'])
        return QtWidgets.QMessageBox.question(self,
                                            "Confirm calculation",
                                            "The following feature channels will be calculated:\n\n{}{}"
                                            .format("\n".join(full_names), "" if not supervox_string else supervox_string),
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    def on_channels_calculated(self, results):
        """
        Once the channels have been calculated, add them to the Feature Channels tab
        :param results: 
        :return: 
        """
        feat_list = []
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
        # Special case 1 - Supervoxel Source Channel
            if not params.get('unique_sv_source', None):
                feat_list.append(fname)
        # Special case 2 - Data
        if 'data' in self.meta_data_result['channel_list']:
            feat_list.append('Data')
        self.predict_widget.use_desc.checkGivenItems(feat_list)

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

        # Disconnect the signal from  the launcher to prevent a never-ending loop
        if self.svx_conn:
            self.launcher.post.disconnect(self.calculate_supervoxels)
            self.svx_conn = False
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
        if self.DM.has_classifier():
            self.predict_widget.predict_btn_widget.btn_predict.setEnabled(True)

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
