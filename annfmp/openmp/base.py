"""
Created on Jun 22, 2020

@author: fgieseke
"""

import time
import numpy
import collections
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA

from .kdtree import KDTree


class ANNFieldPropKDTreeOpenMP:
    def __init__(
        self,
        n_neighbors=3,
        psize=8,
        dim_reduced=5,
        n_subset=1000,
        leaf_size=64,
        n_jobs=-1,
        propagate=True,
        select_best_nn=True,
        float_type="float",
        verbose=1,
        seed=0,
    ):

        self.n_neighbors = n_neighbors
        self.psize = psize
        self.dim_reduced = dim_reduced
        self.n_subset = n_subset
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.propagate = propagate
        self.select_best_nn = select_best_nn
        self.float_type=float_type
        self.verbose = verbose
        self.seed = seed

    def print2dfloat(self, kk, n, arr):

        for i in range(0, n):
            print("[{:.6f}".format(arr[i, 0]), end="")
            for j in range(1, kk):
                print(", {:.6f}".format(arr[i, j]), end="")
            print("]")
        print("]")

    def fit(self, image_a, image_b):
        """
        NOTE:

        image_a and image_b should have shape (h,w,c)
        """

        numpy.random.seed(self.seed)
        
        self.image_a = image_a
        self.image_b = image_b

        self._timers = collections.OrderedDict()

        self._n_cols = self.image_a.shape[1] - self.psize + 1
        self._n_rows = self.image_a.shape[0] - self.psize + 1

        # (1) fit pca model (on subset of the data)
        tstart = time.time()
        self._pca_model = self._fit_pca_model(self.image_a, self.image_b)
        self._timers["pca_fit"] = time.time() - tstart

        # (2) create patches
        tstart = time.time()
        patches_a = self._create_patches(self.image_a)
        patches_b = self._create_patches(self.image_b)
        self._timers["extracting_patches"] = time.time() - tstart

        # (3) apply pca model
        # FIXME: patches are converted to float32/float64 here; can
        # be handled with less memory via C function
        tstart = time.time()
        patches_a_reduced = self._apply_pca(patches_a)
        patches_b_reduced = self._apply_pca(patches_b)
        self._timers["pca_apply"] = time.time() - tstart

        # (4) k-d tree based nearest neighbor propagation
        # initialize nn indices (which are updated incrementally)
        nn_indices = numpy.zeros((patches_a_reduced.shape[0]), dtype=numpy.int32)

        # fit k-d tree
        tstart = time.time()
        self.kdtree = KDTree(
            n_neighbors=self.n_neighbors,
            leaf_size=self.leaf_size,
            splitting_type="longest",
            n_jobs=self.n_jobs,
            float_type=self.float_type,
            verbose=self.verbose,
        )
        self.kdtree.fit(patches_b_reduced)

        self._timers["kdtree_fit"] = time.time() - tstart

        # process remaining rows
        if self.verbose > 0:
            print("Processing rows ...")
        tstart = time.time()

        # since there is just one leaf visit, this must hold
        assert self.leaf_size > self.n_neighbors

        self.kdtree.process_rows(
            nn_indices,
            patches_a,
            patches_b,
            patches_a_reduced,
            self._n_rows,
            self._n_cols,
            self.n_neighbors,
            self.propagate,
            self.select_best_nn,
        )
        self._timers["remaining_rows"] = time.time() - tstart

        if self.verbose > 0:
            print("-----------------------------------------")
            print("--------------- RUNTIMES ----------------")
            print("-----------------------------------------")
            print(
                "Extraction of patches:\t{}".format(self._timers["extracting_patches"])
            )
            print("PCA fit:\t\t{}".format(self._timers["pca_fit"]))
            print("PCA apply:\t\t{}".format(self._timers["pca_apply"]))
            print("K-D Tree (fitting):\t{}".format(self._timers["kdtree_fit"]))
            print("Remaining rows:\t\t{}".format(self._timers["remaining_rows"]))
            print(
                "Total runtime:\t\t{}".format(
                    self._timers["extracting_patches"]
                    + self._timers["pca_fit"]
                    + self._timers["pca_apply"]
                    + self._timers["kdtree_fit"]
                    + self._timers["remaining_rows"]
                )
            )
            print("-----------------------------------------")

        return nn_indices

    def _create_patches(self, img):

        n_channels = img.shape[-1]

        patches = image.extract_patches_2d(img, (self.psize, self.psize))

        if self.verbose > 0:
            print("Patches for image have shape: {}".format(patches.shape))

        patches = patches.reshape((-1, self.psize * self.psize * n_channels))

        if self.verbose > 0:
            print("Reshaped patches for image have shape: {}".format(patches.shape))

        return patches

    def _fit_pca_model(self, image_a, image_b):

        subset_a = image.extract_patches_2d(
            image_a,
            (self.psize, self.psize),
            max_patches=int(self.n_subset / 2),
            random_state=self.seed,
        )

        subset_b = image.extract_patches_2d(
            image_b,
            (self.psize, self.psize),
            max_patches=int(self.n_subset / 2),
            random_state=self.seed + 1,
        )

        subset_a = subset_a.reshape((subset_a.shape[0], -1))
        subset_b = subset_b.reshape((subset_b.shape[0], -1))
        both_subsets = numpy.concatenate((subset_a, subset_b), axis=0)

        model = PCA(n_components=self.dim_reduced)
        model.fit(both_subsets)

        return model

    def _apply_pca(self, patches):

        patches_reduced = self._pca_model.transform(patches).astype(self._get_numpy_float_type())

        if self.verbose > 0:
            print("Patches reduced for image: {}".format(patches_reduced.shape))

        return patches_reduced

    def _get_numpy_float_type(self):

        if self.float_type == "float":
            return numpy.float32
        elif self.float_type == "double":
            return numpy.float64
        
        raise Exception("Unknown float type: {}".format(self.float_type))
    
    @property
    def patches_a(self):
        
        return self._create_patches(self.image_a)

    @property
    def patches_b(self):

        return self._create_patches(self.image_b)        