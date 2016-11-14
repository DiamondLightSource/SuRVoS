

import numpy as np

from sklearn.utils import shuffle
from sklearn.decomposition import IncrementalPCA, PCA, \
                                  DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.cluster import MiniBatchKMeans, KMeans


class Quantizer(object):

    def __init__(self, n_bins=100, n_samples=10000, random_state=None,
                 incremental=False, n_jobs=1, **kwargs):
        self.incremental = incremental
        if self.incremental:
            self.kmeans = MiniBatchKMeans(n_clusters=n_bins,
                                          random_state=random_state, **kwargs)
        else:
            self.kmeans = KMeans(n_clusters=n_bins, random_state=random_state,
                                 n_jobs=n_jobs, **kwargs)
        self.n_samples = n_samples
        self.random_state = random_state
        self.incremental

    def fit(self, X, y=None):
        if self.n_samples is not None and self.n_samples < X.shape[0]:
            X = shuffle(X, random_state=self.random_state)[:self.n_samples]
        self.kmeans.fit(X)

    def partial_fit(self, X, y=None):
        if not self.incremental:
            raise Exception('`partial_fit` not supported if not `incremental`')
        self.kmeans.partial_fit(X)

    def transform(self, X):
        y = self.predict(X)
        return self.kmeans.cluster_centers_[y]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def predict(self, X):
        return self.kmeans.predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


class Textonizer(object):

    def __init__(self, n_textons=100, n_samples=10000, whiten=True,
                 random_state=None, incremental=False, n_jobs=1, **kwargs):
        self.incremental = incremental
        if self.incremental:
            self.kmeans = MiniBatchKMeans(n_clusters=n_textons,
                                          random_state=random_state, **kwargs)
            self.pca = IncrementalPCA(whiten=whiten)
        else:
            self.kmeans = KMeans(n_clusters=n_textons, random_state=random_state,
                                 n_jobs=n_jobs, **kwargs)
            self.pca = PCA(whiten=whiten)
        self.n_samples = n_samples
        self.random_state = random_state
        self.incremental = incremental

    def fit(self, X, y=None):
        if self.n_samples is not None and self.n_samples < X.shape[0]:
            X = shuffle(X, random_state=self.random_state)[:self.n_samples]
        X = self.pca.fit_transform(X)
        self.kmeans.fit(X)

    def partial_fit(self, X, y=None):
        if not self.incremental:
            raise Exception('`partial_fit` not supported if not `incremental`')
        self.pca.partial_fit(X)
        X = self.pca.transform(X)
        self.kmeans.partial_fit(X)

    def transform(self, X):
        y = self.predict(X)
        return self.kmeans.cluster_centers_[y]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        X = self.pca.transform(X)
        return self.kmeans.predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
