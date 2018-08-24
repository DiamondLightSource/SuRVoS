
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans, KMeans, Birch
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import label as cclabel
from skimage.util import img_as_float
from skimage import color
from skimage.segmentation import relabel_sequential
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from .rag import rag_from_neighbors
from .spencoding import spmeans
from ._qpbo import solve_binary, solve_aexpansion
from ._supersegments import _neighbors, _postprocess_labels

class Timer(object):
    def __init__(self, name='Timer'):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        self.tend = (time.time() - self.tstart)
        print('[%s] Elapsed: %.4f seconds' % (self.name, self.tend))

def mean_hue(h1, h2):
    mh = np.zeros_like(h1)
    mask1 = np.abs(h2 - h1) > 180
    mask2 = mask1 & (h1 + h2 < 360.)
    mask3 = mask1 & (h1 + h2 >= 360.)
    mh[mask2] = (h1[mask2] + h2[mask2] + 360.) / 2.
    mh[mask3] = (h1[mask3] + h2[mask3] - 360.) / 2.
    mh[~mask1] = (h1[~mask1] + h2[~mask1]) / 2.
    return mh

def colordiff(A, B):
    if B.shape[0] == 1:
        B = np.repeat(B, A.shape[0], axis=0)
    ac = np.sqrt(A[:, 1]**2 + A[:, 2]**2)
    bc = np.sqrt(B[:, 1]**2 + B[:, 2]**2)
    mc = (ac + bc)/2
    g = (1 - np.sqrt(mc**7 / (mc**7 + 25.0**7))) / 2.
    A = A.copy()
    A[:, 1] *= (1 + g)
    B = B.copy()
    B[:, 1] *= (1 + g)

    A = color.lab2lch(A)
    B = color.lab2lch(B)

    dl = (B[:, 0] - A[:, 0])
    dc = (B[:, 1] - A[:, 1])
    dh = (B[:, 2] - A[:, 2])
    mask1 = (A[:, 1] * B[:, 1] == 0)
    mask2 = (~mask1) & (dh > 180)
    mask3 = (~mask1) & (dh < -180)
    dh[mask1] = 0
    dh[mask2] -= 360
    dh[mask3] += 360

    dh = 2 * np.sqrt(A[:, 1] * B[:, 1]) * np.sin(np.deg2rad(dh / 2.))
    ml = (A[:, 0] + B[:, 0]) / 2
    mc = (A[:, 1] + B[:, 1]) / 2
    mh = np.zeros_like(dh)
    mh[mask1] = A[mask1, 2] + B[mask1, 2]
    mh[~mask1] = mean_hue(A[~mask1, 2], B[~mask1, 2])

    mls = (ml - 50)**2
    sl = 1.0 + 0.015 * mls / np.sqrt(20 + mls)

    # chroma weight
    sc = 1 + 0.045 * mc

    # hue weight
    t = 1 - 0.17 * np.cos(np.deg2rad(mh - 30)) + \
            0.24 * np.cos(np.deg2rad(2 * mh)) + \
            0.32 * np.cos(np.deg2rad(3 * mh + 6)) - \
            0.20 * np.cos(np.deg2rad(4 * mh - 63))
    sh = 1 + 0.015 * mc * t

     # rotation term
    dtheta = 30 * np.exp(-((mh - 275)/25)**2)
    cr = 2 * np.sqrt(mc**7 / (mc**7 + 25.0**7))
    tr = -np.sin(np.deg2rad(2*dtheta)) * cr
    return np.sqrt((dl/(1*sl))**2 + (dc/(1*sc))**2 + (dh/(1*sh))**2 + \
                   tr * (dc/(1*sc)) * (dh/(1*sh)))




def HPC(data, edges, nbow=20, lamda=1, sampling='random', nsamples=10000,
        label_potential='l1', unary_sq=True, online=True, gamma=None, max_iter=5,
        truncated=False, rng=42, return_centers=False, max_size=1000, min_size=10,
        max_neighbors=50, verbose=True, return_image=False, num_clusters=None,
        raw_data_group=False, colordiff=False):

    labels, label_cost, centers = qmrf_regions(data, edges, nbow=nbow, lamda=lamda, sampling='random', nsamples=nsamples,
                                               label_potential=label_potential, unary_sq=unary_sq, online=online,
                                               max_iter=max_iter, gamma=gamma, return_centers=True,
                                               verbose=verbose)

    splabels = postprocess_regions(labels, edges, label_cost, data, max_size=max_size,
                                   min_size=min_size, max_neighbors=max_neighbors,
                                   num_clusters=num_clusters, verbose=verbose,
                                   raw_data_group=raw_data_group)
    if return_centers:
        return splabels, labels, centers
    return splabels


def img2data(image, lab=True, sigma=None):
    if image.shape[-1] in (3, 4) and lab:
        image = color.rgb2lab(image)
    image = img_as_float(image)
    if sigma is not None:
        image = gaussian_filter(image, sigma)

    if image.shape[-1] in (3, 4):
        return image.reshape(-1, image.shape[-1])
    else:
        return image.reshape(-1, 1)


def grid_edges(shape, connectivity=4):
    shape = shape if type(shape) == tuple else shape.shape
    B = np.arange(np.prod(shape[:2]), dtype=np.int32).reshape(shape[:2])
    edges_h = np.dstack([B[:, :-1], B[:, 1:]]).reshape(-1, 2)
    edges_v = np.dstack([B[:-1, :], B[1:, :]]).reshape(-1, 2)
    edges = np.r_[edges_h, edges_v]
    if connectivity == 8:
        edges_ht = np.dstack([B[1:, :-1], B[:-1, 1:]]).reshape(-1, 2)
        edges_hb = np.dstack([B[:-1, :-1], B[1:, 1:]]).reshape(-1, 2)
        edges = np.r_[edges, edges_ht, edges_hb]
    return edges


def cube_edges(image, connectivity=6):
    B = np.arange(np.prod(image.shape[:3]), dtype=np.int32).reshape(image.shape[:3])
    edges_h = np.dstack([B[:, :, :-1], B[:, :, 1:]]).reshape(-1, 2)
    edges_v = np.dstack([B[:, :-1, :], B[:, 1:, :]]).reshape(-1, 2)
    edges_d = np.dstack([B[:-1, :, :], B[1:, :, :]]).reshape(-1, 2)
    edges = np.r_[edges_h, edges_v, edges_d]
    if connectivity == 18:
        edges_1 = np.dstack([B[:, :-1, :-1], B[:, 1:, 1:]]).reshape(-1, 2)
        edges_2 = np.dstack([B[:, 1:, :-1], B[:, :-1, 1:]]).reshape(-1, 2)
        edges_3 = np.dstack([B[:-1, :, :-1], B[1:, :, 1:]]).reshape(-1, 2)
        edges_4 = np.dstack([B[1:, :, :-1], B[:-1, :, 1:]]).reshape(-1, 2)
        edges_5 = np.dstack([B[:-1, :-1, :], B[1:, 1:, :]]).reshape(-1, 2)
        edges_6 = np.dstack([B[1:, :-1, :], B[:-1, 1:, :]]).reshape(-1, 2)
        edges = np.r_[edges, edges_1, edges_2, edges_3, edges_4, edges_5, edges_6]
    if connectivity == 24:
        edges_1 = np.dstack([B[:-1, :-1, :-1], B[1:, 1:, 1:]]).reshape(-1, 2)
        edges_2 = np.dstack([B[1:, :-1, :-1], B[:-1, 1:, 1:]]).reshape(-1, 2)
        edges_3 = np.dstack([B[:-1, 1:, :-1], B[1:, :-1, 1:]]).reshape(-1, 2)
        edges_4 = np.dstack([B[:-1, :-1, 1:], B[1:, 1:, :-1]]).reshape(-1, 2)
        edges = np.r_[edges, edges_1, edges_2, edges_3, edges_4]
    return edges


def qmrf_regions(data, edges, nbow=20, lamda=1, sampling='random', nsamples=10000,
                 label_potential='l1', unary_sq=True, online=True, gamma=None, max_iter=5,
                 truncated=False, rng=42, verbose=True, return_centers=False,
                 return_edge_costs=True):
    with Timer('Colors'):
        if nbow == 'birch':
            clf = Birch(threshold=0.8, branching_factor=100)
        elif online:
            clf = MiniBatchKMeans(n_clusters=nbow, verbose=verbose, random_state=rng,
                                  batch_size=100, max_iter=100, max_no_improvement=10)
        else:
            clf = KMeans(n_clusters=nbow, verbose=verbose, random_state=rng)

        if nsamples is None:
            dist = clf.fit_transform(data)
        else:
            if sampling == 'random':
                idx = np.random.choice(data.shape[0], nsamples, replace=False)
            else:
                n = np.sqrt(nsamples)
                ratio = image.shape[0] / float(image.shape[1])
                ny = int(n * ratio)
                nx = int(n / ratio)
                y = np.linspace(0, image.shape[0], ny, endpoint=False) + (image.shape[0]//ny//2)
                x = np.linspace(0, image.shape[1], nx, endpoint=False) + (image.shape[1]//nx//2)
                xx, yy = np.meshgrid(x, y)
                idx = np.round(yy * image.shape[1] + xx).astype(int).flatten()
            clf.fit(data[idx])
            dist = clf.transform(data)

        if nbow == 'birch':
            centers = clf.subcluster_centers_
        else:
            centers = clf.cluster_centers_

    with Timer('Unary'):
        K = centers.shape[0]

        if label_potential == 'color':
            unary_cost = np.zeros((data.shape[0], centers.shape[0]), np.float32)
            for i in range(centers.shape[0]):
                unary_cost[:, i] = colordiff(data, centers[i:i+1])
        else:
            unary_cost = dist.astype(np.float32)

        if unary_sq:
            unary_cost **= 2

    with Timer('Pairwise'):
        if label_potential == 'l1':
            label_cost = np.abs(centers[:, None, :] - centers[None, ...]).sum(-1)
        elif label_potential == 'l2':
            label_cost = np.sqrt(((centers[:, None, :] - centers[None, ...])**2).sum(-1))
        elif label_potential == 'potts':
            label_cost = np.ones((K, K), int) - np.eye(K, dtype=int)
        elif label_potential == 'color':
            label_cost = np.zeros((centers.shape[0], centers.shape[0]), np.float32)
            for i in range(centers.shape[0]):
                label_cost[:, i] = colordiff(centers, centers[i:i+1])
        if truncated:
            label_cost = np.maximum(1, label_cost)
        label_cost = (label_cost * lamda).astype(np.float32)

    if verbose:
        print("=================")
        print("Minimizing graph:")
        print("Nodes: %d, edges: %d, labels: %d" % \
              (unary_cost.shape[0], edges.shape[0], label_cost.shape[0]))
        print("UnarySq: %s, LabelPotential: %s, EdgeCost: %s" % \
              (unary_sq, label_potential, (gamma is not None)))
        print("#################")

    with Timer('Edge Cost'):
        diff = ((data[edges[:, 0]] - data[edges[:, 1]])**2).sum(axis=1)
        if gamma is not None and type(gamma) in [int, float]:
            edge_costs = np.exp(-gamma * diff).astype(np.float32)
        elif gamma == 'auto':
            edge_costs = np.exp(-diff.mean() * diff).astype(np.float32)
        elif gamma == 'color':
            edge_costs = 1. / (1. + colordiff(data[edges[:, 0]], data[edges[:, 1]]))
            edge_costs = edge_costs.astype(np.float32)
        else:
            edge_costs = np.ones(edges.shape[0], dtype=np.float32)

    with Timer('Minimize'):
        if label_cost.shape[0] == 2:
            labels = solve_binary(edges, unary_cost, edge_costs, label_cost)
        else:
            labels = solve_aexpansion(edges, unary_cost, edge_costs, label_cost)

    if return_centers:
        return labels, label_cost, centers

    return labels, label_cost


def postprocess_regions(labels, edges, label_cost, data, max_size=1000, min_size=None, max_neighbors=50,
                        num_clusters=None, verbose=True, raw_data_group=False):
    neighbors = _neighbors(edges, labels.shape[0], max_neighbors)
    splabels = _postprocess_labels(labels, neighbors, max_size=max_size)

    num_sps = splabels.max()+1

    print("Num Superpixels: %d" % (splabels.max()+1))

    if min_size is not None and min_size > 1:
        counts = np.bincount(splabels)
        rag = rag_from_neighbors(splabels, neighbors)

        for p, d in rag.nodes(data=True):
            d['labels'] = {p}
            d['olabel'] = p

        if verbose:
            print("Num smaller superpixels: %d" % (counts < min_size).sum())

        spedges = np.array(rag.edges(), int)
        if not raw_data_group:
            olabels = spmeans(labels[:, None].astype(np.float32), splabels,
                              splabels.max()+1).astype(np.int32)
            edge_cost = np.ravel(label_cost[olabels[spedges[:, 0]],
                                            olabels[spedges[:, 1]]])
        else:
            costs = spmeans(data.astype(np.float32), splabels, splabels.max()+1)
            edge_cost = ((costs[spedges[:, 0]] - costs[spedges[:, 1]])**2).sum(1)
            edge_cost = edge_cost.astype(np.float32)

        if raw_data_group:
            order = edge_cost + (min(counts[spedges[:, 0]] * counts[spedges[:, 1]]))/ counts.max()
        else:
            order = edge_cost
        idx = np.ravel(np.argsort(order))

        for i in idx:
            p, q = spedges[i]
            if q not in rag.node[p]['labels'] and (counts[p] < min_size or counts[q] < min_size):
                plabels = rag.node[p]['labels']
                qlabels = rag.node[q]['labels']
                pset = list(rag.node[p]['labels'] - set([p, q]))
                qset = list(rag.node[q]['labels'] - set([p, q]))
                counts[pset] += counts[q]
                counts[qset] += counts[p]
                total_nodes = plabels | qlabels
                for i in total_nodes:
                    rag.node[i]['labels'] = total_nodes
                counts[p] += counts[q]
                counts[q] += counts[p]

                num_sps -= 1
                if num_clusters is not None and num_clusters >= num_sps:
                    break

        qlabels = np.arange(len(rag))
        label_map = np.full(qlabels.size, -1, np.int32)
        for ix, (n, d) in enumerate(rag.nodes_iter(data=True)):
            for label in d['labels']:
                if label_map[label] == -1:
                    label_map[label] = ix

        qlabels = label_map[qlabels]
        splabels = qlabels[splabels]

    splabels = relabel_sequential(splabels)[0]
    splabels -= splabels.min()

    if verbose and min_size is not None:
        print("Num Superpixels after merging: %d" % (splabels.max()+1))

    return splabels
