'''
Created on Jun 19, 2020

@author: fgieseke
'''

import time
import numpy
import collections
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class ANNFieldSK():

    def __init__(self,
                 psize=8,
                 dim_reduced=5,
                 n_subset=1000,
                 leaf_size=16,
                 n_jobs=-1,
                 verbose=1,
                 seed=0,
                 ):

        self.psize = psize
        self.dim_reduced = dim_reduced
        self.n_subset = n_subset
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.seed = seed

    def fit(self, image_a, image_b):

        numpy.random.seed(self.seed)
        
        self._timers = collections.OrderedDict()
        
        # (1) create patches
        tstart = time.time()
        self.patches_a, self.patches_b = self._create_patches(image_a, image_b)
        self._timers['extracting_patches'] = time.time() - tstart

        # (2) fit pca model (on subset of the data)
        tstart = time.time()
        self._pca_model = self._fit_pca_model(image_a, image_b)
        self._timers['pca_fit'] = time.time() - tstart

        # (3) apply pca model
        tstart = time.time()
        patches_a_reduced, patches_b_reduced = self._apply_pca(self.patches_a, self.patches_b)
        self._timers['pca_apply'] = time.time() - tstart

        # (4) compute nearest patches
        if self.verbose > 0:
            print("Fitting k-d tree ...")
        tstart = time.time()
        nnmodel = NearestNeighbors(
            n_neighbors=1, 
            algorithm='kd_tree', 
            leaf_size=self.leaf_size, 
            n_jobs=self.n_jobs
        )
        nnmodel.fit(patches_b_reduced)
        self._timers['kdtree_fit'] = time.time() - tstart
        
        tstart = time.time()
        if self.verbose > 0:
            print("Traversing k-d tree ...")
        _, nn_indices = nnmodel.kneighbors(patches_a_reduced)
        
        self._timers['kdtree_traverse'] = time.time() - tstart
        
        if self.verbose > 0:
            print("-----------------------------------------")
            print("--------------- RUNTIMES ----------------")
            print("-----------------------------------------")
            print("Extraction of patches:\t{}".format(self._timers['extracting_patches']))
            print("PCA fit:\t\t{}".format(self._timers['pca_fit']))
            print("PCA apply:\t\t{}".format(self._timers['pca_apply']))
            print("K-D Tree (fitting):\t{}".format(self._timers['kdtree_fit']))
            print("K-D Tree (traverse):\t{}".format(self._timers['kdtree_traverse']))
            print("-----------------------------------------")
        
        # we only have a single nearest neighbor, hence only flattening is needed
        
        return nn_indices.flatten()
    
    def _create_patches(self, image_a, image_b):

        n_channels = image_a.shape[-1]

        patches_a = image.extract_patches_2d(image_a, (self.psize, self.psize))
        patches_b = image.extract_patches_2d(image_b, (self.psize, self.psize))

        if self.verbose > 0:
            print("Patches for image A have shape: {}".format(patches_a.shape))
            print("Patches for image B have shape: {}".format(patches_b.shape))

        patches_a = patches_a.reshape((-1, self.psize * self.psize * n_channels))
        patches_b = patches_b.reshape((-1, self.psize * self.psize * n_channels))

        if self.verbose > 0:
            print("Reshaped patches for image A have shape: {}".format(patches_a.shape))
            print("Reshaped patches for image B have shape: {}".format(patches_b.shape))

        return patches_a, patches_b

    def _fit_pca_model(self, image_a, image_b):
        
        subset_a = image.extract_patches_2d(
            image_a, 
            (self.psize, self.psize),
            max_patches= int(self.n_subset / 2),
            random_state=self.seed
        )

        subset_b = image.extract_patches_2d(
            image_b, 
            (self.psize, self.psize),
            max_patches= int(self.n_subset / 2),
            random_state=self.seed+1
        )
        
        subset_a = subset_a.reshape((subset_a.shape[0], -1))
        subset_b = subset_b.reshape((subset_b.shape[0], -1))
        both_subsets = numpy.concatenate((subset_a, subset_b), axis=0)
                
        model = PCA(n_components=self.dim_reduced)
        model.fit(both_subsets)
        
        return model
#         
#     def _fit_pca(self, patches_a, patches_b):
# 
#         # random subset (1000 instances from each image)
#         patches_a_subset = patches_a[numpy.random.choice(patches_a.shape[0], self.n_subset, replace=False)]
#         patches_b_subset = patches_b[numpy.random.choice(patches_b.shape[0], self.n_subset, replace=False)]
# 
#         if self.verbose > 0:
#             print("Subset of patches created for image A: {}".format(patches_a_subset.shape))
#             print("Subset of patches created for image B: {}".format(patches_b_subset.shape))
# 
#         # fit PCA model
#         if self.verbose > 0:
#             print("Fitting PCA model ...")
# 
#         model = PCA(n_components=self.dim_reduced)
#         both_subsets = numpy.concatenate((patches_a_subset, patches_b_subset), axis=0)
#         model.fit(both_subsets)
# 
#         if self.verbose > 0:
#             print("PCA model fitted!")
# 
#         return model
    
    def _apply_pca(self, patches_a, patches_b):

        patches_a_reduced = self._pca_model.transform(patches_a)
        patches_b_reduced = self._pca_model.transform(patches_b)

        if self.verbose > 0:
            print("Patches reduced for image A: {}".format(patches_a_reduced.shape))
            print("Patches reduced for image B: {}".format(patches_b_reduced.shape))

        return patches_a_reduced, patches_b_reduced    
