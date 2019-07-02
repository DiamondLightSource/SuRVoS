
import h5py as h5
import numpy as np
import logging as log
import ast

import networkx as nx
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import ExtraTreesClassifier, \
                             RandomForestClassifier, \
                             AdaBoostClassifier,\
                             GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans

from sklearn.linear_model import SGDClassifier
from scipy.stats import entropy

from skimage.segmentation import relabel_sequential

from ..lib._spencoding import _sp_labels
from ..lib.spencoding import spmeans, sphist, spstats
from ..lib.spgraph import aggregate_neighbors
from ..lib._qpbo import solve_binary, solve_aexpansion

from ..core import DataModel


DM = DataModel.instance()


def obtain_classifier(clf_p):
    if clf_p['clf'] == 'ensemble':
        mode = 'ensemble'
        if clf_p['type'] == 'rf':
            clf = RandomForestClassifier(n_estimators=clf_p['n_estimators'],
                                         max_depth=clf_p['max_depth'],
                                         n_jobs=clf_p['n_jobs'])
        elif clf_p['type'] == 'erf':
            clf = ExtraTreesClassifier(n_estimators=clf_p['n_estimators'],
                                       max_depth=clf_p['max_depth'],
                                       n_jobs=clf_p['n_jobs'])
        elif clf_p['type'] == 'ada':
            clf = AdaBoostClassifier(n_estimators=clf_p['n_estimators'],
                                     learning_rate=clf_p['learning_rate'])
        else:
            clf = GradientBoostingClassifier(n_estimators=clf_p['n_estimators'],
                                             max_depth=clf_p['max_depth'],
                                             learning_rate=clf_p['learning_rate'],
                                             subsample=clf_p['subsample'])
    elif clf_p['clf'] == 'svm':
        mode = 'svm'
        clf = SVC(C=clf_p['C'], gamma=clf_p['gamma'], kernel=clf_p['kernel'],
                  probability=True)
    elif clf_p['clf'] == 'sgd':
        mode = 'sgd'
        clf = SGDClassifier(loss=clf_p['loss'], penalty=clf_p['penalty'],
                            alpha=clf_p['alpha'], n_iter=clf_p['n_iter'])
    else:
        raise Exception('Classifier not supported')
    return clf, mode


def compute_supervoxel_descriptor(supervoxels, descriptors, desc_type, desc_bins):
    'Mean' 'Quantized' 'Textons' 'Covar' 'Sigma Set'

    if desc_type == 'Mean':
        return spmeans(descriptors, supervoxels)
    elif desc_type == 'Covar':
        return spstats(descriptors, supervoxels, mode='add', norm=None)
    elif desc_type == 'Sigma Set':
        return spstats(descriptors, supervoxels, mode='add', sigmaset=True, norm=None)

    if desc_type == 'Textons':
        log.info('+ Applying PCA')
        descriptors = IncrementalPCA(batch_size=100).fit_transform(descriptors)

    log.info('+ Quantizing descriptors')
    cluster = MiniBatchKMeans(n_clusters=desc_bins, batch_size=100).fit_predict(descriptors)
    return sphist(cluster.astype(np.int32), supervoxels, nbins=desc_bins)


def extract_descriptors(supervoxels=None, features=None,
                        projection=None, desc_type=None, desc_bins=None,
                        nh_order=None, sp_edges=None):
    total = len(features)

    log.info('+ Reserving memory for {} features'.format(total))
    descriptors = np.zeros(DM.region_shape() + (total,), np.float32)

    for i in range(total):
        log.info('    * Loading feature {}'.format(features[i]))
        descriptors[..., i] = DM.load_slices(features[i])

    sp = None
    mask = None

    if supervoxels is not None:
        log.info('+ Loading supervoxels')
        sp = DM.load_slices(supervoxels)
        if sp.min() < 0:
            raise Exception('Supervoxels need to be recomputed for this ROI')
        descriptors.shape = (-1, total)

        num_sv = DM.attr(supervoxels, 'num_supervoxels')
        mask = np.zeros(num_sv, np.bool)
        mask[sp.ravel()] = True

        log.info('+ Computing descriptors: {} ({})'.format(desc_type, desc_bins))
        descriptors = compute_supervoxel_descriptor(sp, descriptors,
                                                    desc_type, desc_bins)
        nh_order = int(nh_order)
        if nh_order > 0:
            log.info('+ Loading edges into memory')
            edges = DM.load_ds(sp_edges)

            log.info('+ Filtering edges for selected ROI')
            idx = mask[edges[:, 0]] & mask[edges[:, 1]]
            edges = edges[idx]

            log.info('+ Aggregating neighbour features')
            G = nx.Graph()
            G.add_edges_from(edges)
            descriptors = aggregate_neighbors(descriptors, G, mode='append',
                                              norm='mean', order=nh_order)
        descriptors = descriptors[mask]

    return descriptors, sp, mask


def predict_proba(y_data=None, p_data=None, train=False,
                  level_params=None, desc_params=None,
                  clf_params=None, ref_params=None,
                  out_labels=None, out_confidence=None):
    log.info("+ Extracting descriptors: {}".format(desc_params))
    X, supervoxels, svmask = extract_descriptors(**desc_params)
    full_svmask = svmask.copy()

    if train:
        clf, mode = obtain_classifier(clf_params)
        log.info("+ Creating classifier: {}".format(clf_params))

        log.info("+ Loading labels")
        labels = DM.load_slices(y_data)
        if level_params['plevel'] is not None:
            parent_labels = DM.load_slices(p_data)
            mask = parent_labels == level_params['plabel']
        else:
            mask = None
    else:
        if DM.has_classifier():
            log.info("+ Getting existing classifier")
            clf = DM.get_classifier_from_model()
        else:
            log.error("For some reason, no previous classifier exists!")
            return
        mask = None

    if supervoxels is not None:
        nsp = DM.attr(desc_params['supervoxels'], 'num_supervoxels')
        log.info('+ Extracting supervoxel labels')
        if train:
            nlbl = DM.attr(y_data, 'label').max() + 1
            labels = _sp_labels(supervoxels.ravel(), labels.ravel(), nsp, nlbl, 0)
            if mask is not None:
                mask = mask.astype(np.int16)
                mask = np.bincount(supervoxels.ravel(), weights=mask.ravel() * 2 - 1) > 0
            if X.shape[0] < labels.shape[0]:  # less supervoxels
                labels = labels[svmask]
                if mask is not None:
                    mask = mask[svmask]
            y = labels
            y.shape = -1

            if mask is not None:
                mask.shape = -1
                idx_train = (y > -1) & mask
            else:
                idx_train = (y > -1)

            X_train = X[idx_train]
            y_train = y[idx_train]

            if clf is not None:
                _train_classifier(clf, X_train, y_train,
                                  project=desc_params['projection'])
                log.info('+ Adding classifier to data model')
                DM.add_classifier_to_model(clf)
            elif clf is None:
                log.error("No Classifier found!")
                return None

    return predict_and_save(X, clf, full_svmask, nsp, out_confidence, out_labels, supervoxels, ref_params, mask, svmask)


def predict_and_save(X, clf, full_svmask, nsp, out_confidence, out_labels, supervoxels, ref_params, mask, svmask):
    result = classifier_predict(X, clf)
    probs = result['probs']
    labels = np.asarray(list(set(np.unique(result['class'][supervoxels])) - set([-1])), np.int32)
    pred = result['class']
    if supervoxels is not None:
        if ref_params['ref_type'] != 'None':
            log.info('+ Remapping supervoxels')
            svmap = np.full(nsp, -1, supervoxels.dtype)
            svmap[full_svmask] = np.arange(svmask.sum())

            log.info('+ Extracting graph')
            refine_lamda = ref_params['lambda']
            edges = DM.load_ds(ref_params['sp_edges'])
            edge_weights = DM.load_ds(ref_params['sp_eweights'])
            mean_weights = edge_weights.mean()
            edge_weights = np.minimum(edge_weights, mean_weights) / mean_weights

            log.info('+ Remapping edges')
            idx = full_svmask[edges[:, 0]] & full_svmask[edges[:, 1]]
            edges = edges[idx]
            edges[:, 0] = svmap[edges[:, 0]]
            edges[:, 1] = svmap[edges[:, 1]]
            edge_weights = edge_weights[idx]

            log.info('  * Unary potentials')
            unary = (-np.ma.log(probs)).filled()
            unary = unary.astype(np.float32)
            mapping = np.zeros(pred.max() + 1, np.int32)
            mapping[labels] = np.arange(labels.size)
            idx = np.where(pred > -1)[0]
            col = mapping[pred[idx]]
            unary[idx, col] = 0
            pairwise = edge_weights.astype(np.float32)

            log.info('  * Pairwise potentials')
            if ref_params['ref_type'] == 'Appearance':
                log.info('+ Extracting descriptive weights')
                dists = np.sqrt(np.sum((X[edges[:, 0]] - X[edges[:, 1]]) ** 2, axis=1))
                pairwise *= np.exp(-dists.mean() * dists)

            log.debug("  * Shapes: {}, {}, {}, {}".format(unary.shape, pairwise.shape,
                                                          edges.shape, edge_weights.shape))
            log.debug("  * Edges: min: {}, max: {}".format(edges.min(0), edges.max(0)))
            log.debug("  * Unary: min: {}, max: {}, mean: {}"
                      .format(unary.min(0), unary.max(0), unary.mean(0)))
            log.debug("  * Pairwise: min: {}, max: {}, mean: {}"
                      .format(pairwise.min(), pairwise.max(), pairwise.mean()))

            log.info('+ Refining labels')
            label_cost = np.ones((labels.size, labels.size), np.float32)
            label_cost[np.diag_indices_from(label_cost)] = 0
            label_cost *= refine_lamda
            if label_cost.shape[0] == 2:
                refined = solve_binary(edges, unary, pairwise, label_cost)
            else:
                refined = solve_aexpansion(edges, unary, pairwise, label_cost)
            pred[mask] = labels[refined[mask]]

            if mask is not None:
                pred[~mask] = -1
                #conf[~mask] = -1
        else:
            pass  # TODO pixel refinement
        log.info('+ Mapping predictions back to pixels')
        pred_map = np.empty(nsp, dtype=result['class'].dtype)
        conf_map = np.empty(nsp, dtype=result['probs'].dtype)
        log.info('+ Measuring uncertainty')
        # Slice each list of confidences with the corresponding predicted class
        # to return the confidence for that class
        conf_result = list(map(lambda x, y: x[y], result['probs'], result['class']))
        pred_map[full_svmask] = result['class']
        conf_map[full_svmask] = conf_result
        pred = pred_map[supervoxels]
        conf = conf_map[supervoxels]
        pred.shape = DM.region_shape()
        conf.shape = DM.region_shape()
    log.info('+ Saving results to disk')
    DM.create_empty_dataset(out_labels, shape=DM.data_shape, dtype=pred.dtype)

    DM.write_slices(out_labels, pred, params=dict(labels=labels, active=True))
    DM.create_empty_dataset(out_confidence, shape=DM.data_shape, dtype=conf.dtype)
    DM.write_slices(out_confidence, conf, params=dict(labels=labels, active=True))
    return out_labels, out_confidence, labels

def _train_classifier(clf, X_train, y_train, rnd=42, project=None):

    if ast.literal_eval(project) is not None:
        log.info('+ Projecting features')
        if project == 'rproj':
            proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=rnd)
        elif project == 'std':
            proj = StandardScaler()
        elif project == 'pca':
            proj = PCA(n_components='mle', whiten=True, random_state=rnd)
        else:
            print(project)
            print(type(project))
            log.error('Projection {} not available'.format(project))
            return

        X_train = proj.fit_transform(X_train)

    log.info('+ Training classifier')
    clf.fit(X_train, y_train)

def classifier_predict(X, clf):
    result = {}
    log.info('+ Predicting labels')
    result['class'] = clf.predict(X)
    result['probs'] = clf.predict_proba(X)
    return result
