

from __future__ import division

import numpy as np
import logging as log

from sklearn.metrics import cohen_kappa_score, jaccard_similarity_score

from ..core import DataModel

DM = DataModel.instance()


def compare_segmentations(levelA=None, levelB=None, labelsA=None, labelsB=None,
                          acc=False, dice=False, jacc=False, cohen=False):
    log.info('+ Loading level A into memory')
    levelA = DM.load_slices(levelA)

    indexes = set()
    result = dict()

    log.info('+ Mapping labels from A')
    mappingA = np.full(levelA.shape, -1, np.int8)
    for s, t in labelsA.items():
        mappingA[levelA[...] == s] = t
        indexes |= set([t])

    log.info('+ Loading level B into memory')
    levelB = DM.load_slices(levelB)

    log.info('+ Mapping labels from B')
    mappingB = np.full(levelB.shape, -1, np.int8)
    for s, t in labelsB.items():
        mappingB[levelB[...] == s] = t
        indexes |= set([t])

    mappingA += 1
    mappingB += 1

    log.info('+ Computing intersection and overlapping')
    overlap = ((mappingA == mappingB) & (mappingA > 0))
    union = ((mappingA > 0) | (mappingB > 0))

    n = max(indexes) + 2
    m = float(mappingA.size)

    log.info('+ Couting target labels')
    inter = (mappingA * overlap).ravel()
    result['countsOverlap'] = np.bincount(inter, minlength=n) / m
    result['countsA'] = np.bincount(mappingA.ravel(), minlength=n) / m
    result['countsB'] = np.bincount(mappingB.ravel(), minlength=n) / m

    log.info('+ Computing scores')
    ninter = overlap.sum()
    nnzA = (mappingA > 0).sum()
    nnzB = (mappingB > 0).sum()

    dA = mappingA[union]
    dB = mappingB[union]

    if acc:
        result['acc'] = ninter / float(nnzB)
    if dice:
        result['dice'] = 2 * ninter / float(nnzA + nnzB)
    if jacc:
        result['jacc'] = jaccard_similarity_score(dB.ravel(), dA.ravel())
    if cohen:
        result['cohen'] = cohen_kappa_score(mappingB.ravel(), mappingA.ravel(),
                                            labels=[i + 1 for i in indexes])

    result['indexes'] = indexes | {-1}

    return result
