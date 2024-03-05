# Test Product Quantization
# References:
#   https://zhuanlan.zhihu.com/p/635665517
#   https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd
#   https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd

from __future__ import annotations

import io
import logging
import unittest

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

BITS2DTYPE = {8: np.uint8}


def load_vectors(fname, max_vectors):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    idx = 0
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))
        idx += 1
        if idx >= max_vectors:
            break
    return idx, d, data


class IndexPQ:
    def __init__(self, dim: int, nr_segs: int, nr_bits: int, **estimator_kwargs):
        if dim % nr_segs != 0:
            raise ValueError("dim needs to be a multiple of nr_segs")

        if nr_bits not in BITS2DTYPE:
            raise ValueError(f"Unsupported number of bits {nr_bits}")

        self.dim = dim
        self.nr_segs = nr_segs
        self.seg_dim = self.dim // self.nr_segs
        self.nr_bits = nr_bits
        self.nr_centroids = 2**self.nr_bits
        self.estimators = [KMeans(n_clusters=self.nr_centroids, **estimator_kwargs) for _ in range(nr_segs)]
        logger.info(f"Creating following estimators: {self.estimators[0]!r}")
        self.is_trained = False
        self.dtype = BITS2DTYPE[self.nr_bits]
        self.dtype_orig = np.float32
        self.codebook = None

    def encode(self, X: np.ndarray):
        n = len(X)
        result = np.empty((n, self.nr_segs), dtype=self.dtype)
        for i in range(self.nr_segs):
            estimator = self.estimators[i]
            X_i = X[:, i * self.seg_dim : (i + 1) * self.seg_dim]
            result[:, i] = estimator.predict(X_i)
        return result

    def train(self, X: np.ndarray):
        if self.is_trained:
            raise ValueError("Training multiple times is not allowed!")

        for i in range(self.nr_segs):
            estimator = self.estimators[i]
            X_i = X[:, i * self.seg_dim : (i + 1) * self.seg_dim]
            logger.info("Fitting KMeans for the {i}-th segment")
            estimator.fit(X_i)

        self.is_trained = True
        self.codes = self.encode(X)

    def search(self, Q: np.ndarray, top_k: int):
        nr_queries = len(Q)
        nr_codes = len(self.codes)
        distance_table = np.empty((nr_queries, self.nr_segs, self.nr_centroids), dtype=self.dtype_orig)
        for i in range(self.nr_segs):
            Q_i = Q[:, i * self.seg_dim : (i + 1) * self.seg_dim]
            centers = self.estimators[i].cluster_centers_
            distance_table[:, i, :] = euclidean_distances(Q_i, centers, squared=True)
        distances = np.zeros((nr_queries, nr_codes), dtype=self.dtype_orig)
        for i in range(self.nr_segs):
            distances += distance_table[:, i, self.codes[:, i]]
        top_k_indices = np.argsort(distances, axis=1)[:, :top_k]
        top_k_distances = np.empty((nr_queries, top_k), dtype=np.float32)
        for i in range(nr_queries):
            top_k_distances[i] = distances[i][top_k_indices[i]]
        return top_k_distances, top_k_indices


class IndexPQTest(unittest.TestCase):
    def setUp(self):
        self.n, self.dim, self.data = load_vectors("/root/data/wiki-news-300d-1M.vec", 1001)
        self.X = []
        self.word2idx = {}
        self.idx2word = {}
        for k, v in self.data.items():
            idx = len(self.X)
            if idx < 1000:
                self.word2idx[k] = idx
                self.idx2word[idx] = k
                self.X.append(v)
            else:
                self.Q = np.array([v])
                self.expected_words = [k]
        self.X = np.array(self.X)
        self.nr_segs = 10
        self.index_pq = IndexPQ(self.dim, self.nr_segs, 8)

    def test_search(self):
        self.index_pq.train(self.X)
        distances, indices = self.index_pq.search(self.X, 1)
        expected_indices = np.array([i for i in range(self.n)])
        # np.testing.assert_equal(np.squeeze(indices), expected_indices)

        distances, indices = self.index_pq.search(self.Q, 5)
        for indice in np.squeeze(indices):
            print(indice, self.idx2word[indice])
        print(self.expected_words)
