

import numpy as np
from .spencoding import normalize


def aggregate_neighbors(data, graph, mode='add', weighted=False, norm='l1', order=1):
    if mode not in ['append', 'add']:
        raise Exception('Only `append` or `add` methods are accepted')

    ndata = np.zeros_like(data)
    neighbors = np.zeros(data.shape[0])

    for node in graph:
        nh = graph.neighbors(node)
        neighbors[node] = len(nh)
        if not weighted:
            ndata[node] = data[nh].sum(axis=0)
        else:
            weights = np.array([graph[node][nb]['boundary'] for nb in nh], float)
            ndata[node] = (data[nh] * weights[:, None]).sum(axis=0)

        if order == 2:
            currents = nh + [node]
            for nhnode in nh:
                nhnh = filter(lambda x: None if x in currents else x,
                              graph.neighbors(nhnode))
                if len(nhnh) == 0:
                    continue
                currents += nhnh
                neighbors[node] += len(nhnh)
                if not weighted:
                    ndata[node] += data[nhnh].sum(axis=0)
                else:
                    weights = np.array([graph[nhnode][nb]['boundary'] for nb in nhnh], float)
                    ndata[node] += (data[nhnh] * weights[:, None]).sum(axis=0)

    if mode == 'add':
        ndata += data
        neighbors += 1

    if norm == 'mean':
        ndata /= neighbors[:, None]
    else:
        ndata = normalize(ndata, norm=norm)

    if mode == 'append':
        ndata = np.c_[data, ndata]

    return ndata
