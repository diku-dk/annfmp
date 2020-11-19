"""
Created on Jun 22, 2020

@author: fgieseke
"""

import time
import collections
import numpy
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA

try:
    from . import wrapper_float
except Exception as e:
    print("Could not import Swig object: {}".format(str(e)))


class ANNFieldPropKDTreeOpenCLOPT:
    def __init__(
        self,
        n_neighbors=3,
        psize=8,
        dim_reduced=5,
        n_subset=1000,
        leaf_size=64,
        n_jobs=-1,
        select_best_nn=True,
        verbose=1,
        seed=0,
        platform_id=0,
        device_id=0,
    ):

        self.n_neighbors = n_neighbors
        self.psize = psize
        self.dim_reduced = dim_reduced
        self.n_subset = n_subset
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.select_best_nn = select_best_nn
        self.verbose = verbose
        self.seed = seed
        self.platform_id = platform_id
        self.device_id = device_id

    def fit(self, image_a, image_b):

        assert self.select_best_nn == True

        self._timers = collections.OrderedDict()

        tstart = time.time()
        self._pca_model = self._fit_pca_model(image_a, image_b)
        self._timers["pca_fit"] = time.time() - tstart
        # print("pca fit")
        # print(self._timers['pca_fit'])
        # (needed for the swig module)

        tstart = time.time()
        self.image_a = numpy.ascontiguousarray(image_a).astype(numpy.int32)
        self.image_b = numpy.ascontiguousarray(image_b).astype(numpy.int32)

        # columns and rows
        self._n_cols = self.image_a.shape[0] - self.psize + 1
        self._n_rows = self.image_a.shape[1] - self.psize + 1

        # number of patches
        n_patches_a = self._n_cols * self._n_rows

        # array that should contain the output after the propagation phase
        # (it contains, for each patch, the K nearest neighbor indices)
        nn_indices = numpy.zeros((n_patches_a), dtype=numpy.int32)
        nn_distances = numpy.zeros((n_patches_a), dtype=numpy.float32)
        self._timers["python_overhead"] = time.time() - tstart

        # print("comps: ", self._pca_model.components_.shape)
        # print(self._pca_model.components_.astype(numpy.float32))
        # print("means")
        # print(self._pca_model.mean_.astype(numpy.float32))
        # print("futhark")

        self.wrapper_futhark_ctxinp = self._get_wrapper_module().FUTHARK_CTX_INP()
        self._get_wrapper_module().init_extern(
            self.wrapper_futhark_ctxinp, self.dim_reduced,
        )

        # self._get_wrapper_module().fit_extern( self.wrapper_futhark_ctxinp, 1 )
        tstart = time.time()

        self._get_wrapper_module().pair_init(
            self.wrapper_futhark_ctxinp,
            self.image_a,
            self.image_b,
            numpy.ascontiguousarray(self._pca_model.components_).astype(numpy.float32),
            numpy.ascontiguousarray(self._pca_model.mean_).astype(numpy.float32),
            numpy.ascontiguousarray(nn_indices),
            numpy.ascontiguousarray(nn_distances),
            self.n_neighbors,
            self.psize,
            self.dim_reduced,
            self.n_subset,
            self.leaf_size,
            self.seed,
            self.platform_id,
            self.device_id,
        )
        self._get_wrapper_module().fit_extern(self.wrapper_futhark_ctxinp, 0)
        self._get_wrapper_module().pair_free(self.wrapper_futhark_ctxinp)

        self._timers["futhark"] = time.time() - tstart

        self._get_wrapper_module().free_extern(self.wrapper_futhark_ctxinp)

        if self.verbose > 0:
            print("-----------------------------------------")
            print("--------------- RUNTIMES ----------------")
            print("-----------------------------------------")
            print("PCA fit:\t\t{}".format(self._timers["pca_fit"]))
            print("Python overhead:\t{}".format(self._timers["python_overhead"]))
            print("Futhark:\t\t{}".format(self._timers["futhark"]))
            print(
                "Total runtime:\t\t{}".format(
                    self._timers["pca_fit"]
                    + self._timers["python_overhead"]
                    + self._timers["futhark"]
                )
            )
            print("-----------------------------------------")

        return nn_indices.flatten()

    def _get_wrapper_module(self):
        """ Returns the corresponding swig
        wrapper module.

        Returns
        -------
        wrapper : object
            The wrapper object
        """

        return wrapper_float

    @property
    def patches_a(self):

        n_channels = self.image_a.shape[-1]

        patches = image.extract_patches_2d(self.image_a, (self.psize, self.psize))

        patches = patches.reshape((-1, self.psize * self.psize * n_channels))

        return patches

    @property
    def patches_b(self):

        n_channels = self.image_b.shape[-1]

        patches = image.extract_patches_2d(self.image_b, (self.psize, self.psize))

        patches = patches.reshape((-1, self.psize * self.psize * n_channels))

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
