#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp


def _neighbors(int[:, ::1] edges,
               int n_nodes, int max_neighbors):
    cdef int n_edges = edges.shape[0]
    cdef int[:, ::1] neighbors = np.full((n_nodes, max_neighbors), -1, np.int32)
    cdef int[::1] currents = np.zeros(n_nodes, np.int32)
    cdef n, p, q, pc, qc

    for n in range(n_edges):
        p = edges[n, 0]
        q = edges[n, 1]
        pc = currents[p]
        qc = currents[q]
        neighbors[p, pc] = q
        neighbors[q, qc] = p
        currents[p] += 1
        currents[q] += 1

    return np.asarray(neighbors)


def _postprocess_labels(int[::1] segments,
                        int[:, ::1] neighbors,
                        int max_size):
    """ Helper function to remove small disconnected regions from the labels
    Parameters
    ----------
    segments : 1D array of int, shape (NSP,)
        The label field/superpixels.
    neighbors : 2D array of int, shape (NSP, MAX_NH)
        The connexion of neighbors
    max_size: int
        Maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # new object with connected segments initialized to -1
    cdef int[::1] connected_segments = -1 * np.ones_like(segments, dtype=np.int32)
    cdef int current_new_label = 0
    cdef int label = 0

    # variables for the breadth first search
    cdef int current_segment_size = 1
    cdef int bfs_visited = 0
    cdef int[::1] coord_list = np.zeros(max_size, dtype=np.int32)

    cdef int p, q, k
    cdef int n_segments = segments.shape[0]
    cdef int max_neighbors = neighbors.shape[1]


    with nogil:
        for p in range(n_segments):
            if connected_segments[p] >= 0:
                continue

            adjacent = 0
            label = segments[p]
            connected_segments[p] = current_new_label
            current_segment_size = 1
            bfs_visited = 0
            coord_list[bfs_visited] = p

            while bfs_visited < current_segment_size < max_size:
                p = coord_list[bfs_visited]
                for k in range(max_neighbors):
                    q = neighbors[p, k]
                    if q < 0:
                        continue

                    if segments[q] == label and connected_segments[q] == -1:
                        connected_segments[q] = current_new_label
                        coord_list[current_segment_size] = q
                        current_segment_size += 1
                        if current_segment_size >= max_size:
                            break
                bfs_visited += 1

            current_new_label += 1

    return np.asarray(connected_segments)
