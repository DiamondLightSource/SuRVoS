
from ..qt_compat import QtGui, QtCore

import numpy as np
import logging as log
from collections import deque

import os
import ast
import glob
import h5py as h5
from pickle import dumps, loads

from scipy.stats import gaussian_kde

from .singleton import Singleton


@Singleton
class DataModel(QtCore.QObject):

    roi_changed = QtCore.pyqtSignal()
    select_channel = QtCore.pyqtSignal(str)
    update_channel = QtCore.pyqtSignal(str)
    cropped = QtCore.pyqtSignal()
    vmin_changed = QtCore.pyqtSignal(float)
    vmax_changed = QtCore.pyqtSignal(float)
    evmin_changed = QtCore.pyqtSignal(float)
    evmax_changed = QtCore.pyqtSignal(float)
    channel_computed = QtCore.pyqtSignal(str, dict)
    channel_removed = QtCore.pyqtSignal(str)
    feature_updated = QtCore.pyqtSignal(str)
    classifier_trained = QtCore.pyqtSignal(int)
    voxel_descriptor_computed = QtCore.pyqtSignal()
    voxel_descriptor_removed = QtCore.pyqtSignal()
    supervoxel_descriptor_computed = QtCore.pyqtSignal()
    supervoxel_descriptor_removed = QtCore.pyqtSignal()
    supervoxel_descriptor_cleared = QtCore.pyqtSignal()
    level_predicted = QtCore.pyqtSignal(int)
    clf_channel_computed = QtCore.pyqtSignal(int, str, str, bool, dict)

    Axial = 0
    Sagittal = 1
    Coronal = 2

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.main_window = None

        # tmp_data
        self.tmp_data = None
        self.tmp_stats = None

        # Storage
        self.wspath = None
        self.current_idx = 0
        self.selected_gpu = -1

        # DataFiles
        self.data = None
        self.data_shape = None
        self.active_roi = slice(None), slice(None), slice(None)

        # Superpixels
        self.svlabels = None
        self.svindex = None
        self.svtable = None
        self.svtotal = None

        # Megavoxels
        self.mvlabels = None
        self.mvindex = None
        self.mvtable = None
        self.mvtotal = None

        # Classifier
        self.clf = None
        self.clf_name = None
        self.clf_saved_name = None
        self.clf_params = None
        self.X_train = None
        self.y_train = None

        # Parameters to be saved with classifier
        self.desc_params = None

        # GT Labels
        self.gtlevel = 0
        self.gtradius = 1
        self.gtinterpolation = 'linear'
        self.gtselected = None
        self.last_changes = deque(maxlen=10)
        self.redo_changes = deque(maxlen=10)
        self.growing_bbox = ['data', 1., [10, 50, 50], 0]

        # KDE plots
        self.computed_histograms = dict()
        self.channel_computed.connect(self.disable_histogram)

    def disable_histogram(self, name):
        if name in self.computed_histograms:
            del self.computed_kde[name]

    def get_histogram(self, dataset):
        if dataset in self.computed_histograms:
            return self.computed_histograms[dataset]

        log.info('+ Loading channel into memory')
        data = self.load_ds(dataset).ravel()
        attr = self.attrs(dataset)
        amin = attr['vmin']
        amax = attr['vmax']

        log.info('+ Computing Histogram')
        y, x = np.histogram(data[~np.isnan(data)], 10000)
        x = (x[:-1] + x[1:]) / 2.

        self.computed_histograms[dataset] = (x, y)
        return (x, y)

    ##########################################################################
    # TOI Transformation
    ##########################################################################

    def region_shape(self):
        return tuple(map(lambda x: x.stop - x.start, self.active_roi))

    def transform_slices(self, slice_z=None, slice_y=None, slice_x=None):
        slices = []

        for t, s in zip(self.active_roi, (slice_z, slice_y, slice_x)):
            start = t.start
            stop = t.stop

            try:
                s = int(s)
                start += s
                stop = start + 1
            except:
                if s is None:
                    s = slice(None)
                if s.stop is not None:
                    stop = t.start + s.stop
                if s.start is not None:
                    start += s.start

            slices.append(slice(start, stop))

        return slices

    ##########################################################################
    # DATASETS IN / OUT
    ##########################################################################

    def create_empty_dataset(self, dataset, shape, dtype, check=True,
                             params=None, fillvalue=-1):
        ds_file = self.ds_path(dataset)
        ds_folder = os.path.dirname(ds_file)
        if not os.path.exists(ds_folder):
            os.mkdir(ds_folder)
        if check and os.path.isfile(ds_file):
            return False
        with h5.File(ds_file, 'w') as f:
            ds = f.create_dataset('data', shape=shape, fillvalue=fillvalue,
                                  dtype=dtype)
            if params is not None:
                for k, v in params.items():
                    ds.attrs[k] = v
        return True

    def write_dataset(self, dataset, data, params=None):
        ds_file = self.ds_path(dataset)
        ds_folder = os.path.dirname(ds_file)
        if not os.path.exists(ds_folder):
            os.mkdir(ds_folder)
        with h5.File(ds_file, 'a') as f:
            if 'data' in f:
                f['data'].write_direct(data)
            else:
                f.create_dataset('data', data=data)
            if params is not None:
                for k, v in params.items():
                    f['data'].attrs[k] = v

    def load_ds(self, dataset):
        ds_file = self.ds_path(dataset)
        with h5.File(ds_file, 'r') as f:
            data = np.zeros_like(f['data'])
            f['data'].read_direct(data)
        return data

    def load_slices(self, dataset, slice_z=None, slice_y=None, slice_x=None,
                    apply_roi=True):
        ds_file = self.ds_path(dataset)
        if apply_roi:
            sz, sy, sx = self.transform_slices(slice_z, slice_y, slice_x)
        else:
            sz, sy, sx = slice_z, slice_y, slice_x
        with h5.File(ds_file, 'r') as f:
            data = f['data'][sz, sy, sx]
        return data

    def write_slices(self, dataset, data, slice_z=None, slice_y=None,
                     slice_x=None, params=None, apply_roi=True):
        ds_file = self.ds_path(dataset)
        if apply_roi:
            sz, sy, sx = self.transform_slices(slice_z, slice_y, slice_x)
        else:
            sz, sy, sx = slice_z, slice_y, slice_x
        with h5.File(ds_file, 'a') as f:
            f['data'][sz, sy, sx] = data
            if params is not None:
                for k, v in params.items():
                    f['data'].attrs[k] = v
        return data

    def remove_dataset(self, dataset):
        ds_file = self.ds_path(dataset)
        if os.path.exists(ds_file):
            os.remove(ds_file)

    def add_training_data(self, X_train, y_train):
        """
        Adds data used to train classifier to the data model
        :param X_train: Values
        :param y_train: Labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def has_training_data(self):
        return self.X_train is not None and self.y_train is not None

    def get_training_data(self):
        """
        :return: A tuple with the training data and labels
        """
        if self.has_training_data():
            return self.X_train, self.y_train

    def new_annotations_added(self, y_data):
        data = self.load_slices(y_data)
        return np.any(data > -1)

    def save_classifier(self, path, lbl_attrs):
        """
        Save classifier to filepath
        :param path: filepath for classifier output
        """
        if self.has_classifier():
            clf_pkl = dumps(self.clf)
            with h5.File(path, 'w') as out_file:
                if self.has_training_data():
                    out_file['X_train'] = self.X_train
                    out_file['y_train'] = self.y_train
                out_file['classifier'] = np.void(clf_pkl)
                out_file['classifier'].attrs['class'] = type(self.clf).__name__
                out_file['classifier'].attrs['features'] = self.desc_params.get('features')
                out_file['classifier'].attrs['parameters'] = str(self.clf_params)
                # Add the label information
                self.write_attrs_to_empty_dataset(out_file, "lbl_info", lbl_attrs)
                # Copy the supervoxel information across
                svds = 'supervoxels/supervoxels'
                supervox_attrs = self.attrs(svds)
                self.write_attrs_to_empty_dataset(out_file, "supervox_info", supervox_attrs)
                # Get the metadata for the feature channels
                result_list = self.get_channel_metadata()
                # Create the datasets
                for result in result_list:
                    dataset_name = "{}_{}".format(result.get('feature_idx', ''), result.get('feature_type', 'data'))
                    # fix cases where data is used as an input..
                    if dataset_name == "_data":
                        dataset_name = dataset_name[1:]
                    self.write_attrs_to_empty_dataset(out_file, dataset_name, result)
        else:
            log.error("Data Model has no classifier!")

    def write_attrs_to_empty_dataset(self, file, dataset, attrs_dict):
        """
        Helper function to create and empty dataset in a file and 
        populate HDF5 dataset attrs
        :param file: H5py file object
        :param dataset: Name of the dataset to create (str)
        :param attrs_dict: Dictionary of attributes to add to dataset
        """
        file.create_dataset(dataset, dtype='f')
        for k, v in attrs_dict.items():
            file[dataset].attrs[k] = v

    def load_saved_settings_from_file(self, path):
        """
        Returns a dictionary containing all the metadata for settings saved along with the classifier
        :param path: Path to classifier HDF5 file
        :return: Dictionary with metadata
        """
        result = {}
        if os.path.exists(path):
            with h5.File(path, 'r') as in_file:
                try:
                    # Get the list of channels used by the classifier
                    channel_list = in_file['classifier'].attrs['features']
                    # Store dicts with the metadata for each channel in the list
                    for channel_string in channel_list:
                        # channel_string is in form 'channel/name'
                        name = channel_string.split('/')[1]
                        result[name] = dict(in_file[name].attrs)
                        result.setdefault('channel_list', []).append(name)
                        result[name]['out'] = channel_string
                    result['lbl_attrs'] = dict(in_file['lbl_info'].attrs.items())
                    result['sv_attrs'] = dict(in_file['supervox_info'].attrs.items())
                except Exception as e:
                    log.error("Could not load attributes: {}".format(e))
            if "data" in result:  # Unfortunate special case - data has been used for prediction
                result['data']['feature_name'] = "data"
                result['data']['feature_-dx'] = -1
            return result
        else:
            log.error("File does not appear to exist")

    def get_classifier_params(self):
        return self.clf_params

    def get_channel_metadata(self):
        """
        Get the required metadata from each feature channel file
        :return metadata: list of dictionaries, each containing data for one channel
        """
        return [self.attrs(feature) for feature in self.desc_params.get('features')]

    def add_classifier_to_model(self, clf, clf_params):
        """
        Add classifier to model object along with parameters
        :param clf: classifier to add to object
        """
        self.clf = clf
        self.clf_name = clf.__class__.__name__
        self.clf_params = clf_params


    def get_classifier_from_model(self):
        """
        Retrieves classifier from model
        :return: classifier
        """
        if not self.clf:
            log.error("No Classifier in model")
        else:
            return self.clf

    def has_classifier(self):
        return self.clf is not None

    def load_classifier(self, path):
        if os.path.exists(path):
            with h5.File(path, 'r') as in_file:
                try:
                    clf_str = in_file['classifier']
                    clf_params = ast.literal_eval(in_file['classifier'].attrs['parameters'])
                    clf = loads(clf_str[()].tostring())
                    class_str =  in_file['classifier'].attrs['class']
                    X_train = in_file["X_train"][()]
                    y_train = in_file["y_train"][()]
                    self.add_training_data(X_train, y_train)
                except Exception as e:
                    log.error(path, e)
                    return False

            if str(class_str) == type(clf).__name__:
                self.add_classifier_to_model(clf, clf_params)
                path = os.path.normpath(path)
                split_path = path.split(os.sep)
                self.clf_saved_name = " - ".join(split_path[-3::2])
                log.info("Classifier {} Loaded: {}".format(self.clf_saved_name, type(self.clf)))
                return True
            else:
                log.error("Classifier not loaded. Class description {} does not"
                          " match classifier type {}".format(str(class_str), type(clf).__name__))
                return False
        else:
            log.error("File does not appear to exist")
            return False

    def add_desc_params_to_model(self, desc_params):
        """
        Add desc_params to model object
        :param desc_params: dictionary of descriptors
        """
        self.desc_params = desc_params

    ##########################################################################
    # ATTRIBUTES AND MEMBERSHIP
    ##########################################################################

    def set_attrs(self, dataset, attrs):
        ds_file = self.ds_path(dataset)
        with h5.File(ds_file, 'a') as f:
            for k, v in attrs.items():
                if isinstance(v, list): #this helps with python 3 compatibility
                    if len(v)!=0:
                        if isinstance(v[0], str) or isinstance(v[0], bytes):
                            v = [np.string_(x) for x in v]
                f['data'].attrs[k] = v

    def attrs(self, dataset):
        ds_file = self.ds_path(dataset)
        with h5.File(ds_file, 'r') as f:
            attr = dict(f['data'].attrs)
        return attr

    def attr(self, dataset, attr, value=None):
        ds_file = self.ds_path(dataset)
        with h5.File(ds_file, 'r') as f:
            if attr in f['data'].attrs:
                if value is None:
                    val = f['data'].attrs[attr]
                else:
                    val = (f['data'].attrs[attr] == value)
            else:
                val = None
        return val

    def ds_shape(self, dataset):
        ds_file = self.ds_path(dataset)
        with h5.File(ds_file, 'r') as f:
            shape = f['data'].shape
        return shape

    def has_ds(self, dataset):
        active = False
        ds_file = self.ds_path(dataset)
        if os.path.isfile(ds_file):
            with h5.File(ds_file, 'r') as f:
                if 'data' in f and 'active' in f['data'].attrs:
                    active = f['data'].attrs['active']
        return active

    def has_grp(self, group):
        ds_folder = self.grp_path(group)
        return os.path.exists(ds_folder)

    def ds_path(self, dataset):
        if dataset.startswith('/'):
            dataset = dataset[1:]
        return os.path.join(self.wspath, '{}.h5'.format(dataset))

    def grp_path(self, group):
        return os.path.join(self.wspath, group)

    ##########################################################################
    # SCAN UTILITIES
    ##########################################################################

    def scan_datasets_group(self, group, shape=None, dtype=None, path=''):
        datasets = []
        for name, ds in group.items():
            curr_path = '{}/{}'.format(path, name)
            if hasattr(ds, 'shape'):
                if len(ds.shape) == 3 \
                        and (shape is None or ds.shape == shape) \
                        and (dtype is None or ds.dtype == dtype):
                    datasets.append(curr_path)
            else:
                extra = self.scan_datasets_group(ds, shape=shape, path=curr_path)
                if len(extra) > 0:
                    datasets += extra
        return datasets


    def available_hdf5_datasets(self, path, shape=None, dtype=None):
        datasets = []
        with h5.File(path, 'r') as f:
            datasets = self.scan_datasets_group(f, shape=shape, dtype=dtype)
        return datasets

    ##########################################################################
    # FEATURES AND DESCRIPTORS
    ##########################################################################

    def find_dataset(self, group, filter_active=True):
        grp_dir = self.grp_path(group)
        ds = glob.glob('{}/*.h5'.format(grp_dir))
        ds = ['{}/{}'.format(group, os.path.basename(d[:-3])) for d in ds]
        if filter_active:
            ds = [d for d in ds if self.attr(d, 'active') == True]
        return ds

    def available_channels(self, return_names=False, filter_active=True):
        if self.has_grp('channels'):
            possible = self.find_dataset('channels', filter_active=filter_active)
            af = []
            for f in possible:
                attrs = self.attrs(f)
                if 'active' in attrs and 'feature_idx' in attrs:
                    fidx = attrs['feature_idx']
                    if return_names:
                        af.append((fidx, f, attrs))
                    else:
                        af.append((fidx, f))
        else:
            af = []
        return af

    def available_annotations(self, return_names=False):
        if self.has_grp('annotations'):
            possible = self.find_dataset('annotations')
            af = []
            for f in possible:
                attrs = self.attrs(f)
                if 'active' in attrs and 'levelid' in attrs:
                    af.append((attrs['levelid'], f, attrs['active']))
        else:
            af = []
        return af
