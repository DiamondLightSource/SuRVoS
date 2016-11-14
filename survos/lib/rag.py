

import numpy as np
import networkx as nx

from ._rag import _create_rag_2d, _create_rag_3d
from skimage.segmentation import mark_boundaries

from skimage import measure, draw

from sklearn.metrics.pairwise import distance_metrics, kernel_metrics
from ._dist import bhattacharya, intersection


def edge_weight(x, y, mode='rbf', gamma=0.5):
    dists = distance_metrics()
    kernels = kernel_metrics()
    kernels['bhattacharya'] = bhattacharya
    kernels['intersection'] = intersection
    if mode in dists:
        diff = dists[mode](x, y)
    elif mode in kernels:
        diff = kernels[mode](x, y, gamma=gamma)
    else:
        raise Exception('Mode not recognised')

    return np.float64(diff)


def is_distance(mode):
    return mode in distance_metrics()


def min_weight(graph, src, dst, n):
    """Callback to handle merging nodes by choosing minimum weight.
    Returns either the weight between (`src`, `n`) or (`dst`, `n`)
    in `graph` or the minimum of the two when both exist.
    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The verices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.
    Returns
    -------
    weight : float
        The weight between (`src`, `n`) or (`dst`, `n`) in `graph` or the
        minimum of the two when both exist.
    """

    # cover the cases where n only has edge to either `src` or `dst`
    default = {'weight': np.inf}
    w1 = graph[n].get(src, default)['weight']
    w2 = graph[n].get(dst, default)['weight']
    return min(w1, w2)


class WRAG(nx.Graph):

    """
    The Region Adjacency Graph (RAG) of an image, subclasses
    `networx.Graph <http://networkx.github.io/documentation/latest/reference/classes.graph.html>`_
    """

    def __init__(self, data=None, **attr):
        super(WRAG, self).__init__(data, **attr)
        try:
            self.max_id = max(self.nodes_iter())
        except ValueError:
            # Empty sequence
            self.max_id = 0
        self.dist_weights = None

    def merge_nodes(self, src, dst, weight_func=min_weight, in_place=True,
                    extra_arguments=[], extra_keywords={}):
        """Merge node `src` and `dst`.
        The new combined node is adjacent to all the neighbors of `src`
        and `dst`. `weight_func` is called to decide the weight of edges
        incident on the new node.
        Parameters
        ----------
        src, dst : int
            Nodes to be merged.
        weight_func : callable, optional
            Function to decide edge weight of edges incident on the new node.
            For each neighbor `n` for `src and `dst`, `weight_func` will be
            called as follows: `weight_func(src, dst, n, *extra_arguments,
            **extra_keywords)`. `src`, `dst` and `n` are IDs of vertices in the
            RAG object which is in turn a subclass of
            `networkx.Graph`.
        in_place : bool, optional
            If set to `True`, the merged node has the id `dst`, else merged
            node has a new id which is returned.
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `weight_func`.
        extra_keywords : dictionary, optional
            The dict of keyword arguments passed to the `weight_func`.
        Returns
        -------
        id : int
            The id of the new node.
        Notes
        -----
        If `in_place` is `False` the resulting node has a new id, rather than
        `dst`.
        """
        src_nbrs = set(self.neighbors(src))
        dst_nbrs = set(self.neighbors(dst))
        neighbors = (src_nbrs | dst_nbrs) - set([src, dst])

        if in_place:
            new = dst
        else:
            new = self.next_id()
            self.add_node(new)

        for neighbor in neighbors:
            w = weight_func(self, src, new, neighbor, *extra_arguments,
                            **extra_keywords)
            self.add_edge(neighbor, new, weight=w)

        self.node[new]['labels'] = (self.node[src]['labels'] +
                                    self.node[dst]['labels'])
        self.remove_node(src)

        if not in_place:
            self.remove_node(dst)

        return new

    def add_node(self, n, attr_dict=None, **attr):
        """Add node `n` while updating the maximum node id.
        .. seealso:: :func:`networkx.Graph.add_node`."""
        super(RAG, self).add_node(n, attr_dict, **attr)
        self.max_id = max(n, self.max_id)

    def add_edge(self, u, v, attr_dict=None, **attr):
        """Add an edge between `u` and `v` while updating max node id.
        .. seealso:: :func:`networkx.Graph.add_edge`."""
        super(RAG, self).add_edge(u, v, attr_dict, **attr)
        self.max_id = max(u, v, self.max_id)

    def copy(self):
        """Copy the graph with its max node id.
        .. seealso:: :func:`networkx.Graph.copy`."""
        g = super(RAG, self).copy()
        g.max_id = self.max_id
        return g

    def next_id(self):
        """Returns the `id` for the new node to be inserted.
        The current implementation returns one more than the maximum `id`.
        Returns
        -------
        id : int
            The `id` of the new node to be inserted.
        """
        return self.max_id + 1

    def _add_node_silent(self, n):
        """Add node `n` without updating the maximum node id.
        This is a convenience method used internally.
        .. seealso:: :func:`networkx.Graph.add_node`."""
        super(RAG, self).add_node(n)

    def set_edge_weights(self, X, mode='rbf', gamma=0.5):
        for p, q, d in self.edges_iter(data=True):
            d['weight'] = edge_weight(X[p], X[q], mode=mode, gamma=gamma)
        self.dist_weights = is_distance(mode)


def rag_from_neighbors(nodes, neighbors, min_boundary=None,
                       norm_counts='unit', margin=0, return_rag=True):
    """Creates a Region Adjacency Graph between superpixels.

    Parameters
    ----------
    splabels : 2D or 3D ndarray
        Input superpixel labels for a 2D or 3D image.
    connectivity : int, optional
        Order of the neighbours, which must have a value of ${1, 2}$.
        For a 2D image `1` corresponds to 4-connected pixels while `2`
        corresponds to 8-connected pixels. For a 3D image `1` and `2`
        correspond to 6-connected and 18-connected voxels respectively.
    min_boundary : int, optional
        Minimum number of boundary pixels that a pair of superpixels
        need to share in order to consider them neighbours.
    norm_counts : string, optional
        Normalize the number of boundary pixels that any pair of
        superpixels share. Possible values are 'unit' which performs
        unit normalization as `counts /= counts.max()` and
        `margin` which performs `counts = np.max(counts, margin) / margin`.
    margin : int, optional
        An heuristic number of boundary pixels required to consider two
        neighbouring superpixels *perfect neighbours*. Used only if
        `norm_counts='margin'`.

    Returns
    -------
    graph : RAG (subclass of networkx.Graph)
        An undirected graph containing all the superpixels as nodes
        and the connections between them as edges. The number of
        boundary pixels that a pair of superpixels share is returned
        as an edge property: `boundary`.

    """
    n_nodes = np.int64(nodes.max() + 1)

    nodes = np.tile(nodes, neighbors.shape[1])
    neighbors = neighbors.flatten('f')
    neighbors[neighbors > -1] = nodes[neighbors[neighbors > -1]]

    idx = (neighbors != -1) & (neighbors != nodes)
    nodes = nodes[idx]
    neighbors = neighbors[idx]

    idx = nodes > neighbors
    nodes[idx], neighbors[idx] = neighbors[idx], nodes[idx]

    crossing_hash = nodes + neighbors.astype(np.int64) * n_nodes
    if min_boundary is not None:
        unique_hash, counts = np.unique(crossing_hash, return_counts=True)
    else:
        unique_hash = np.unique(crossing_hash)

    neighbors = np.c_[unique_hash % n_nodes, unique_hash // n_nodes]
    neighbors = neighbors.astype(np.int32)

    if min_boundary is not None:
        idx = (counts >= min_boundary)
        neighbors = neighbors[idx]
        counts = counts[idx]
        if norm_counts == 'unit':
            counts /= float(counts.max())
        elif norm_counts == 'margin':
            counts = np.minimum(counts, margin) / float(margin)
    else:
        counts = np.ones(neighbors.shape[0])

    # Create Region Adjacency Graph
    if return_rag:
        graph = WRAG()
        graph.add_nodes_from(np.arange(n_nodes))
        graph.add_weighted_edges_from(np.c_[neighbors, counts], weight='boundary')
        return graph

    return neighbors


def create_rag(splabels, connectivity=1, min_boundary=None,
               norm_counts='unit', margin=0, return_rag=True,
               return_counts=True):
    """Creates a Region Adjacency Graph between superpixels.

    Parameters
    ----------
    splabels : 2D or 3D ndarray
        Input superpixel labels for a 2D or 3D image.
    connectivity : int, optional
        Order of the neighbours, which must have a value of ${1, 2}$.
        For a 2D image `1` corresponds to 4-connected pixels while `2`
        corresponds to 8-connected pixels. For a 3D image `1` and `2`
        correspond to 6-connected and 18-connected voxels respectively.
    min_boundary : int, optional
        Minimum number of boundary pixels that a pair of superpixels
        need to share in order to consider them neighbours.
    norm_counts : string, optional
        Normalize the number of boundary pixels that any pair of
        superpixels share. Possible values are 'unit' which performs
        unit normalization as `counts /= counts.max()` and
        `margin` which performs `counts = np.max(counts, margin) / margin`.
    margin : int, optional
        An heuristic number of boundary pixels required to consider two
        neighbouring superpixels *perfect neighbours*. Used only if
        `norm_counts='margin'`.

    Returns
    -------
    graph : RAG (subclass of networkx.Graph)
        An undirected graph containing all the superpixels as nodes
        and the connections between them as edges. The number of
        boundary pixels that a pair of superpixels share is returned
        as an edge property: `boundary`.

    """
    n_labels = splabels.max() + 1

    if (splabels.ndim == 2 and connectivity not in (4, 8)) or \
       (splabels.ndim == 3 and connectivity not in (6, 18, 26)):
        raise Exception('Only {1, 2} values are supported for `connectivity`')

    if splabels.ndim == 2:
        nodes, neighbors = _create_rag_2d(splabels, connectivity)
    else:
        nodes, neighbors = _create_rag_3d(splabels, connectivity)

    nodes = np.tile(nodes, connectivity//2)
    neighbors = neighbors.flatten('f')

    idx = (neighbors != -1) & (neighbors != nodes)
    nodes = nodes[idx]
    neighbors = neighbors[idx]

    idx = nodes > neighbors
    nodes[idx], neighbors[idx] = neighbors[idx], nodes[idx]

    n_nodes = np.int64(splabels.max()+1)
    crossing_hash = nodes + neighbors.astype(np.int64) * n_nodes
    if min_boundary is not None or return_counts:
        unique_hash, counts = np.unique(crossing_hash, return_counts=True)
    else:
        unique_hash = np.unique(crossing_hash)

    neighbors = np.c_[unique_hash % n_nodes, unique_hash // n_nodes]
    neighbors = neighbors.astype(np.int32)

    if min_boundary is not None:
        idx = (counts >= min_boundary)
        neighbors = neighbors[idx]
        counts = counts[idx]

    if return_counts:
        if norm_counts == 'unit':
            counts /= float(counts.max())
        elif norm_counts == 'margin':
            counts = np.minimum(counts, margin) / float(margin)

    # Create Region Adjacency Graph
    if return_rag:
        graph = WRAG()
        graph.add_nodes_from(np.arange(n_labels))
        if return_counts:
            graph.add_weighted_edges_from(np.c_[neighbors, counts], weight='boundary')
        else:
            graph.add_edges_from(neighbors)
        return graph
    elif return_counts:
        return neighbors, counts
    else:
        return neighbors



def draw_rag(rag, splabels, img, border_color=(1,1,0),
             node_color=(0,0,1), edge_color=(0,1,0)):
    """Draws Region Adjacency Graph's nodes and edges in an image.

    Parameters
    ----------
    rag : RAG (subclass of networkx.Graph)
        Input graph of superpixels (i.e. created by `create_rag`).
    splabels : 2D ndarray
        Input superpixel labels for a 2D grayscale or color image.
    img : 2D or 3D ndarray
        2D grayscale or color input image.
    border_color : tuple of 3 int
        Color of the superpixel boundaries.
    node_color : tuple of 3 int
        Color of the nodes' centroids.
    edge_color : tuple of 3 int
        Color of edges between connected nodes.

    Returns
    -------
    out : color image, same shape as input `img`
        Returns the input image `img` with superpixel boundaries
        and graph's nodes and edges overlayed on top of it.

    """

    assert splabels.ndim == 2, 'Only 2D (+color) images are accepted'

    if img.ndim == 2:
        img = color.gray2rgb(img)

    regions = measure.regionprops(splabels+1)

    for n, region in enumerate(regions):
        rag.node[n]['centroid'] = region['centroid']

    if border_color is not None:
        img = mark_boundaries(img, splabels, color=border_color,
                              outline_color=None)

    for n1, n2, data in rag.edges_iter(data=True):
        r1, c1 = map(int, rag.node[n1]['centroid'])
        r2, c2 = map(int, rag.node[n2]['centroid'])

        circle = draw.circle(r1, c1, 2)
        img[circle] = node_color

        rr, cc = draw.line(r1, c1, r2, c2)
        img[rr, cc] = edge_color

    return img
