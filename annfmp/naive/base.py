'''
Created on Jun 19, 2020

@author: fgieseke
'''

import time
import numpy
import collections
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA

from .kdtree import KDTree

class ANNFieldPropKDTreeNaive():

    def __init__(self,
                 n_neighbors=3,
                 psize=8,
                 dim_reduced=5,
                 n_subset=1000,
                 leaf_size=16,
                 n_jobs=-1,
                 select_best_nn=True,
                 verbose=1,
                 seed=0
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

    def fit(self, image_a, image_b):
        
        numpy.random.seed(self.seed)

        self._timers = collections.OrderedDict()
        
        self._n_cols = image_a.shape[0] - self.psize + 1
        self._n_rows = image_a.shape[1] - self.psize + 1
                
        # (1) create patches
        tstart = time.time()
        self.patches_a, self.patches_b = self._create_patches(image_a, image_b)
        self._timers['extracting_patches'] = time.time() - tstart

        # (2) fit pca model (on subset of the data)
        tstart = time.time()
        self._pca_model = self._fit_pca(self.patches_a, self.patches_b)
        self._timers['pca_fit'] = time.time() - tstart

        # (3) apply pca model
        tstart = time.time()
        patches_a_reduced, patches_b_reduced = self._apply_pca(self.patches_a, self.patches_b)
        self._timers['pca_apply'] = time.time() - tstart

        # (4) k-d tree based nearest neighbor propagation
        
        # initialize nn indices (which are updated incrementally)
        nn_indices = numpy.zeros((patches_a_reduced.shape[0], self.n_neighbors), dtype=numpy.int32)

        # fit k-d tree
        tstart = time.time()
        self.kdtree = KDTree(self.leaf_size, verbose=self.verbose)
        self.kdtree.fit(patches_b_reduced)
        print(self.kdtree.split_values[:10])
        self._timers['kdtree_fit'] = time.time() - tstart

        # process first row
        if self.verbose > 0:
            print("Processing first row ...")
        tstart = time.time()

#        first_row = patches_a_reduced[:self._n_cols, :]
#         print(first_row[0])
#         print(first_row[1])
#         print(first_row[1000])
#         print(first_row[1001])
#         print(first_row.shape)
                
        self._process_first_row(nn_indices, patches_a_reduced, self._n_cols)
        self._timers['first_row'] = time.time() - tstart
        
#         
#         print(nn_indices[0, :])
#         print(nn_indices[1, :])
#         print(nn_indices[1000, :])
#         print(nn_indices[1001, :])
        print(self._timers['first_row'])        
        
        # process remaining rows
        if self.verbose > 0:
            print("Processing remaining rows ...")
        tstart = time.time()
        self._process_remaining_rows(nn_indices, patches_a_reduced, self._n_rows, self._n_cols)
        self._timers['remaining_rows'] = time.time() - tstart
                

            
        tstart = time.time()
        if self.select_best_nn:  
            
            dists = []
            for n in range(self.n_neighbors):
                diff = (self.patches_a.astype(numpy.float32) - self.patches_b[nn_indices[:,n]].astype(numpy.float32))
                norms = numpy.linalg.norm(diff, axis=1)                
                dists.append(norms)
            diffs = numpy.array(dists).T
            
            best_ns = numpy.argmin(diffs,axis=1)    
            inds = []
            for i in range(len(nn_indices)):
                inds.append(nn_indices[i][best_ns[i]])
            nn_indices = numpy.array(inds)
            
        else:
            
            # only take the single nearest neighbor for each pixel
            nn_indices = nn_indices[:,0].flatten()
        
        self._timers['best_nn_indices'] = time.time() - tstart

        if self.verbose > 0:
            print("-----------------------------------------")
            print("--------------- RUNTIMES ----------------")
            print("-----------------------------------------")
            print("Extraction of patches:\t{}".format(self._timers['extracting_patches']))
            print("PCA fit:\t\t{}".format(self._timers['pca_fit']))
            print("PCA apply:\t\t{}".format(self._timers['pca_apply']))
            print("K-D Tree (fitting):\t{}".format(self._timers['kdtree_fit']))
            print("First row:\t\t{}".format(self._timers['first_row']))
            print("Remaining rows:\t\t{}".format(self._timers['remaining_rows']))
            print("Best nn indices:\t{}".format(self._timers['best_nn_indices']))
            print("-----------------------------------------")
                    
                    
        return nn_indices
    
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
    
    def _fit_pca(self, patches_a, patches_b):

        # random subset (1000 instances from each image)
        patches_a_subset = patches_a[numpy.random.choice(patches_a.shape[0], self.n_subset, replace=False)]
        patches_b_subset = patches_b[numpy.random.choice(patches_b.shape[0], self.n_subset, replace=False)]

        if self.verbose > 0:
            print("Subset of patches created for image A: {}".format(patches_a_subset.shape))
            print("Subset of patches created for image B: {}".format(patches_b_subset.shape))

        # fit PCA model
        if self.verbose > 0:
            print("Fitting PCA model ...")

        model = PCA(n_components=self.dim_reduced)
        both_subsets = numpy.concatenate((patches_a_subset, patches_b_subset), axis=0)
        model.fit(both_subsets)

        if self.verbose > 0:
            print("PCA model fitted!")

        return model
    
    def _apply_pca(self, patches_a, patches_b):

        patches_a_reduced = self._pca_model.transform(patches_a)
        patches_b_reduced = self._pca_model.transform(patches_b)

        if self.verbose > 0:
            print("Patches reduced for image A: {}".format(patches_a_reduced.shape))
            print("Patches reduced for image B: {}".format(patches_b_reduced.shape))

        return patches_a_reduced, patches_b_reduced    

    def _process_first_row(self, nn_indices, patches_a_reduced, n_cols):

        points = patches_a_reduced[:n_cols, :]

        for i in range(len(points)):
            _, nn_indices[i] = self.kdtree.find(points[i], self.n_neighbors)

    def _process_remaining_rows(self, nn_indices, patches_a_reduced, n_rows, n_cols):

        elapsed_time_other = 0
        elapsed_time_propagate = 0
        elapsed_time_brute = 0

        for patch_y in range(1, n_rows):

            start = time.time()
            print("Processing row {} of {} ...".format(patch_y, n_rows))

            points_row = patches_a_reduced[n_cols * patch_y:n_cols * patch_y + n_cols, :]

            # leaf indices stores the indices of the kd tree that need to be processed
            # via brute force at the end of each row traversal
            leaf_indices = numpy.zeros((n_cols, 1 + self.n_neighbors), dtype=numpy.int32)

            # get first leaf index via tree traversal (without backtracking)
            for i in range(n_cols): 
                leaf_idx = self.kdtree.find_first_leaf(points_row[i])           
                leaf_indices[i, 0] = leaf_idx
            end = time.time()
            elapsed_time_other += end - start

            start = time.time()
            # propagate self.n_neighbors leaf indices
            for i in range(n_cols):
                prop_indices = self._propagate_patches(i, patch_y, n_cols, nn_indices, self.kdtree.inverse_lookup)
                leaf_indices[i,1:] = numpy.array(prop_indices).astype(numpy.int32)
            end = time.time()
            elapsed_time_propagate += end - start

            # brute force 
            start = time.time()
            for i in range(n_cols):
                nn_indices[n_cols * patch_y + i] = self._brute_force_group(points_row[i], leaf_indices[i], self.kdtree.leaves, self.n_neighbors)[1]
            end = time.time()
            elapsed_time_brute += end - start   
            
    def _propagate_patches(self, patch_x, patch_y, n_cols, nn_indices, inverse_lookup):
    
        leaf_indices = []
    
        # propagate n_neighbors from the different positions
        for k in range(nn_indices.shape[1]):
    
            # index of patch index "above"
            location_above = n_cols * (patch_y - 1) + patch_x
            best_index_of_patch_above = nn_indices[location_above, k]
    
            prop_index = best_index_of_patch_above
            
            # 'Reverse' the spatial traversal we have done to get propagation nearest neighbor patch
            # additionally checking for out of bounds
            if prop_index + n_cols < len(inverse_lookup):
                prop_index = prop_index + n_cols
    
            # retrieve the leaf of the tree that this patch belongs to
            leaf_index = inverse_lookup[prop_index]
    
            # add it to our brute force pool
            leaf_indices.append(leaf_index)
    
        return leaf_indices
    
    def _brute_force_group(self, patch, leaves_index, leaves, k_neighbors):
        
        points, indices = leaves[leaves_index[0]]
    
        for i in leaves_index[1:]:
            new_points, new_indices = leaves[i]
            points = numpy.vstack((points, new_points))
            indices = numpy.append(indices, new_indices)
    
        diffs = numpy.sum((points - patch)**2, axis=1)**0.5
        neighbor_candidates = numpy.vstack((diffs, indices))
        neighbor_candidates = neighbor_candidates[:, neighbor_candidates[0,:].argsort()][:, :k_neighbors]
    
        return neighbor_candidates             