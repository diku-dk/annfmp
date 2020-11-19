'''
Created on Jun 20, 2020

@author: fgieseke
'''

import numpy

class KDTree():

    def __init__(self, leaf_size=50, verbose=0):

        self.leaf_size = leaf_size
        self.verbose = verbose

    def fit(self, X):

        self.max_depth = int(numpy.log2(len(X) / self.leaf_size)) + 1

        # (2) initialize the internal nodes of the k-d tree
        # we assume that we have perfectly balanced trees (i.e, that we stop at a leaf size > 1)
        self.split_values = numpy.zeros((2 ** self.max_depth) - 1, dtype=numpy.float32)
        self.split_dimensions = numpy.zeros((2 ** self.max_depth) - 1, dtype=numpy.int32)
    
        # (3) initialize the leaves of the k-d tree
        # - list of Numpy arrays containing the points
        # - inverse loop-up for points (point -> leaf index)
        self.leaves = [None] * (2 ** (self.max_depth - 1))
        self.inverse_lookup = numpy.zeros(X.shape[0], dtype=numpy.int32)
    
        # (4) create copy of the original patch indices (which are processed in the same way as the points)
        self.indices = numpy.arange(X.shape[0])
    
        # (5) build k-d tree recursively
        self._make_kd_tree_recursive(X,
            self.indices,
            0,
            0
        )

    def find(self, x, n_neighbors):

        best_neighbors = None
        best_neighbors = self._traverse_tree_backtrack(
                                        x,
                                        0,
                                        best_neighbors,
                                        n_neighbors
                          )

        distances = best_neighbors[0]
        indices = best_neighbors[1]

        return distances, indices

    def find_first_leaf(self, x):

        return self._traverse_tree_no_backtrack(x, 0)
        
    def _make_kd_tree_recursive(self,
                                X,
                                indices,
                                depth,
                                index):
    
        # if depth not reached: create internal node
        if depth < self.max_depth - 1:
    
            # (1) select dimension to split on
            dim = self._select_split_dimension(X)
    
            # (2) sort points and indices w.r.t. dimension dim
            # (find median of the points w.r.t. dimension dim)
            sort_indices = X[:, dim].argsort()
            indices = indices[sort_indices]
            X = X[sort_indices]
            median_idx = (X.shape[0]) // 2
    
            # (3) the split value used
            split_value = X[median_idx][dim]
    
            # (4) store information in internal node (index is the index of the internal node)
            self.split_values[index] = split_value
            self.split_dimensions[index] = dim
    
            # (5) recursive calls
            self._make_kd_tree_recursive(X[: median_idx],
                                         indices[: median_idx],
                                         depth + 1,
                                         (index + 1) * 2 - 1
                                         )
            self._make_kd_tree_recursive(X[median_idx:],
                                         indices[median_idx:],
                                         depth + 1,
                                         (index + 1) * 2
                                         )
    
        # create a leaf
        else:
    
            leaf_index = index + 1 - (2 ** (self.max_depth - 1))
            self.leaves[leaf_index] = [X, indices]
    
            # store inverse loopup indices
            for i in indices:
                self.inverse_lookup[i] = leaf_index    
                
    def _select_split_dimension(self, X, max_rows=100):
    
        # sample just random X rows for fast tree building
        if ((max_rows > 0) and (X.shape[0] < max_rows)):
            row_indices = numpy.random.randint(X.shape[0], size=max_rows)
            X = X[row_indices, :]
    
        best_difference = 0
        best_split = 0
    
        # iterate over all dimensions
        for i in range(X.shape[1]):
    
            # compute the difference between the max and min w.r.t. dimension i
            difference = numpy.max(X[:, i]) - numpy.min(X[:, i])
    
            # update difference and best dimension if needed
            if difference >= best_difference:
    
                best_difference = difference
                best_split = i
    
        return best_split                
    
    def _traverse_tree_backtrack(self, x, node_index, best_neighbors, k_neighbors, alpha=1.0):
    
        # if we have reached a leaf (here, (2 ** (tree_depth - 1)) - 1 corresponds to the max internal node index)
        if node_index >= (2 ** (self.max_depth - 1)) - 1:
    
            # do not continue
            best_neighbors = self._brute_force(x, node_index + 1 - (2 ** (self.max_depth - 1)), best_neighbors, k_neighbors)
    
            return best_neighbors
    
        # else: go left
        if x[self.split_dimensions[node_index]] <= self.split_values[node_index]:
    
            first = (node_index + 1) * 2 - 1
            second = (node_index + 1) * 2
    
        # or go right
        else:
    
            first = (node_index + 1) * 2
            second = (node_index + 1) * 2 - 1
    
        # handle "first" node (just go down)
        best_neighbors = self._traverse_tree_backtrack(x, first, best_neighbors, k_neighbors)
    
        # if backtracking is active, traverse also the "second" node
        if abs(self.split_values[node_index] - x[self.split_dimensions[node_index]]) < (best_neighbors[0][-1] / alpha):
            best_neighbors = self._traverse_tree_backtrack(x, second, best_neighbors, k_neighbors)
    
        return best_neighbors
    
    def _brute_force(self, patch, leaves_index, best_neighbors, k_neighbors):
    
        points, indices = self.leaves[leaves_index]
        diffs = numpy.sum((points - patch) ** 2, axis=1) ** 0.5
    
        neighbor_candidates = numpy.vstack((diffs, indices))
    
        if best_neighbors is not None:
            neighbor_candidates = numpy.hstack((neighbor_candidates, best_neighbors))
    
        neighbor_candidates = neighbor_candidates[:, neighbor_candidates[0, :].argsort()]
        neighbor_candidates = neighbor_candidates[:, :k_neighbors]
    
        return neighbor_candidates  
    
    def _traverse_tree_no_backtrack(self, query, node_index):

        if node_index >= (2 ** (self.max_depth - 1)) - 1:
            return node_index + 1 - (2 ** (self.max_depth - 1))
    
        if query[self.split_dimensions[node_index]] <= self.split_values[node_index]:
            ne = (node_index + 1) * 2 - 1
        else:
            ne = (node_index + 1) * 2
    
        return self._traverse_tree_no_backtrack(query, ne)
              
