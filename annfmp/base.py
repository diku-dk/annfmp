"""
Created on Jun 19, 2020

@author: fgieseke
"""

from .sklearn import ANNFieldSK
from .naive import ANNFieldPropKDTreeNaive
from .openmp import ANNFieldPropKDTreeOpenMP
from .openclopt import ANNFieldPropKDTreeOpenCLOPT


class ANNField:
    def __init__(
        self,
        n_neighbors=1,
        psize=8,
        dim_reduced=24,
        n_subset=1000,
        algorithm="sklearn",
        leaf_size=64,
        n_jobs=-1,
        propagate=True,
        select_best_nn=True,
        verbose=1,
        seed=0,
    ):

        self.n_neighbors = n_neighbors
        self.psize = psize
        self.dim_reduced = dim_reduced
        self.n_subset = n_subset
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.propagate = propagate
        self.select_best_nn = select_best_nn
        self.verbose = verbose
        self.seed = seed

    def get_params(self, deep=True):
        """ Returns the parameters of the model
        """

        params = {
            "n_neighbors": self.n_neighbors,
            "psize": self.psize,
            "dim_reduced": self.dim_reduced,
            "n_subset": self.n_subset,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "n_jobs": self.n_jobs,
            "propagate": self.propagate,
            "select_best_nn": self.select_best_nn,
            "verbose": self.verbose,
            "seed": self.seed,
        }

        return params

    def set_params(self, **parameters):
        """ Sets the parameters of the model
        """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, image_a, image_b):

        if self.algorithm not in ["sklearn", "naive", "openmp", "openclopt"]:
            raise Exception("Unknown algorithm '{}'!".format(self.algorithm))

        if self.algorithm == "sklearn":

            self._model = ANNFieldSK(
                psize=self.psize,
                dim_reduced=self.dim_reduced,
                n_subset=self.n_subset,
                leaf_size=self.leaf_size,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                seed=self.seed,
            )
            nn_indices = self._model.fit(image_a, image_b)

        elif self.algorithm == "naive":

            self._model = ANNFieldPropKDTreeNaive(
                n_neighbors=self.n_neighbors,
                psize=self.psize,
                dim_reduced=self.dim_reduced,
                n_subset=self.n_subset,
                leaf_size=self.leaf_size,
                n_jobs=self.n_jobs,
                select_best_nn=self.select_best_nn,
                verbose=self.verbose,
                seed=self.seed,
            )
            nn_indices = self._model.fit(image_a, image_b)

        elif self.algorithm == "openmp":

            self._model = ANNFieldPropKDTreeOpenMP(
                n_neighbors=self.n_neighbors,
                psize=self.psize,
                dim_reduced=self.dim_reduced,
                n_subset=self.n_subset,
                leaf_size=self.leaf_size,
                n_jobs=self.n_jobs,
                propagate=self.propagate,
                select_best_nn=self.select_best_nn,
                verbose=self.verbose,
                seed=self.seed,
            )
            nn_indices = self._model.fit(image_a, image_b)

        elif self.algorithm == "openclopt":

            self._model = ANNFieldPropKDTreeOpenCLOPT(
                n_neighbors=self.n_neighbors,
                psize=self.psize,
                dim_reduced=self.dim_reduced,
                n_subset=self.n_subset,
                leaf_size=self.leaf_size,
                n_jobs=self.n_jobs,
                select_best_nn=self.select_best_nn,
                verbose=self.verbose,
                seed=self.seed,
            )
            nn_indices = self._model.fit(image_a, image_b)

        return nn_indices

    @property
    def patches_a(self):
        return self._model.patches_a

    @property
    def patches_b(self):
        return self._model.patches_b
