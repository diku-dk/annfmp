/*
 * base.c
 *
 * Copyright (C) 2013-2020 Fabian Gieseke <fabian.gieseke@uni-muenster.de>
 * License: GPL v2
 *
 */
#include "include/base.h"

/**
 * Initializes the *params struct with the parameters provided.
 *
 * @param n_neighbors The number of nearest neighbors to be found
 * @param tree_depth The tree depth of the tree to be built
 * @param max_leaves The maximum number of leaf visits for each query.
 * @param num_threads The number of threads that should be used
 * @param splitting_type The splitting type that can be used during the construction of the tree
 * @param verbosity_level The verbosity level (0==no output, 1==more output, 2==...)
 * @param *params Pointer to struct that is used to store all parameters
 *
 */
void init_extern(int n_neighbors,
		int tree_depth,
		int max_leaves,
		int num_threads,
		int splitting_type,
		int verbosity_level,
		KD_TREE_PARAMETERS *params) {

	set_default_parameters(params);

	params->n_neighbors = n_neighbors;
	params->tree_depth = tree_depth;
	params->max_leaves = max_leaves;
	params->num_threads = num_threads;
	params->verbosity_level = verbosity_level;
	params->splitting_type = splitting_type;

	check_parameters(params);

	omp_set_num_threads(params->num_threads);

}

/**
 * Builds a k-d-tree
 *
 * @param *Xtrain Pointer to array of type "FLOAT_TYPE" (either "float" or "double")
 * @param nXtrain Number of rows in *X (i.e., points/patterns)
 * @param dXtrain Number of columns in *X (one column per point/pattern)
 * @param *kdtree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 */
void fit_extern(FLOAT_TYPE *Xtrain,
		int nXtrain,
		int dXtrain,
		KD_TREE_RECORD *kdtree_record,
		KD_TREE_PARAMETERS *params) {

	kd_tree_init_tree_record(kdtree_record, params->tree_depth, Xtrain, nXtrain, dXtrain);
	kd_tree_generate_training_patterns_indices(kdtree_record);
	kd_tree_build_tree(kdtree_record, params);

//	int i;
//	for (i=0; i<nXtrain; i++){
//		printf("%i ", kdtree_record->points_leaf_indices[i]);
//	}
//	printf("\n");
}

/**
 * Interface (extern): Computes the k nearest neighbors for a given set of test points
 * stored in *Xtest and stores the results in two arrays *distances and *indices.
 *
 * @param *Xtest Pointer to the set of query/test points (stored as FLOAT_TYPE)
 * @param nXtest The number of query points
 * @param dXtest The dimension of each query point
 * @param *distances The distances array (FLOAT_TYPE) used to store the computed distances
 * @param ndistances The number of query points
 * @param ddistances The number of distance values for each query point
 * @param *indices Pointer to arrray storing the indices of the k nearest neighbors for each query point
 * @param nindices The number of query points
 * @param dindices The number of indices comptued for each query point
 * @param *kdtree_record Pointer to struct storing all relevant information for model
 * @param *params Pointer to struct containing all relevant parameters
 *
 */
void neighbors_extern(FLOAT_TYPE * Xtest,
		int nXtest,
		int dXtest,
		FLOAT_TYPE * distances,
		int ndistances,
		int ddistances,
		int *indices,
		int nindices,
		int dindices,
		KD_TREE_RECORD *kdtree_record,
		KD_TREE_PARAMETERS *params) {

	int i, j;
	int K = params->n_neighbors;
	int max_leaves = params->max_leaves;

	// simply parallelize over queries
#pragma omp parallel for
	for (i = 0; i < nXtest; i++) {
		FLOAT_TYPE *tpattern = Xtest + i * dXtest;
		kd_tree_query_tree_sequential(tpattern, 
									  distances + i * K,
									  indices + i * K, 
									  K,
									  max_leaves, 
									  0,
									  kdtree_record);
	}

	char *XI = kdtree_record->XI;
	int size_elt = dXtest * sizeof(FLOAT_TYPE) + sizeof(int);

	// create array containing the original indices
	// (cannot be parallelized due to interleaved access)
	for (i = 0; i < nXtest; i++) {
		for (j = 0; j < K; j++) {

			int index_in_tree = indices[i * K + j];
			indices[i * K + j] = *((int *) (XI + index_in_tree * size_elt + dXtest * sizeof(FLOAT_TYPE)));
			distances[i * K + j] = sqrt(distances[i * K + j]);

		}
	}

}




void print2Dint(int32_t kk, int32_t n, int* arr) {
	printf("\nPriniting integer 2D array: [\n");
	for(int32_t i=0; i<n; i++) {
		printf("[%i", arr[i*kk]);
		for(int32_t j=1; j<kk; j++) {
			printf(", %i", arr[i*kk+j]);
		}
		printf("]\n");
	}
	printf("]\n");
}


void process_rows_extern(
		int * finalindices,
		int nfinalindices, // 1517009
		uint8_t * patchesA,
		int npatchesA,
		int dpatchesA,
		uint8_t * patchesB,
		int npatchesB,
		int dpatchesB,
		FLOAT_TYPE * patchesred,
		int npatchesred, // 1517009
		int dpatchesred, // 4
		int n_rows, // 793
		int n_cols, // 1913
		int n_neighbors, // 3
		int propagate,
		int select_best_nn,
		KD_TREE_RECORD *kdtree_record,
		KD_TREE_PARAMETERS *params){

	/*
	 * COMMENT:
	 *
	 * The patches array contains the patches in a "row-order", that is, the first n_cols
	 * elements of the array correspond to the first row of the original image and so on...
	 *
	 * The n_rows and n_cols are computed based on the original image
	 *  self._n_cols = image_a.shape[1] - self.psize + 1
     *  self._n_rows = image_a.shape[0] - self.psize + 1
	 *
	 */
	int i, j, k, patch_y;
	FLOAT_TYPE *patch;

	// temporary arrays (user per row): indices and distances
	int *indices_local = (int*) malloc(n_cols * n_neighbors * sizeof(int));
	int *indices = (int*) malloc(npatchesred * n_neighbors * sizeof(int));

	FLOAT_TYPE *dists_local = (FLOAT_TYPE*) malloc(n_cols * n_neighbors * sizeof(FLOAT_TYPE));

	// temporary array (used per row): leaf indices stores the indices of the
	// kd tree that need to be processed via brute force at the end of each row traversal
	int *leaf_indices = (int*) malloc(n_cols * (1 + n_neighbors) * sizeof(int));

	// (I) process first row
	printf("Processing first row ...\n");
	// NOTE: The indices here are w.r.t. to rearranged patterns (not the original ones)
#pragma omp parallel for
	for (i=0; i < n_cols; i++){

		kd_tree_query_tree_sequential(
				patchesred + i*dpatchesred,
				dists_local + i * n_neighbors,
				indices_local + i * n_neighbors,
				n_neighbors,
				params->max_leaves,
				0,
				kdtree_record
		);

		// copy local indices to global array
		for (k=0; k < n_neighbors; k++){
			indices[i * n_neighbors + k] = indices_local[i*n_neighbors + k];
		}

	}

	//printf("COSMIN exact knn indices:\n");
	//print2Dint(n_neighbors, 8, indices);

	// (II) process remaining rows
	printf("Processing remaining rows ...\n");

	// fixed so far
	int max_leaves = 1;

	// iterate over all rows
	for (patch_y=1; patch_y < n_rows; patch_y++){

		// pointer to the patches of the current row
		FLOAT_TYPE *patches_row = patchesred + n_cols * patch_y * dpatchesred;

		// (1) first guess via first leaf search
#pragma omp parallel for
		for (i=0; i < n_cols; i++){

			// only the first leaf index is returned; dists_local and indices_local are not used here ...
			patch = patches_row + i * dpatchesred;

			// compute the first leaf index via k-d tree traversal (top to bottom)
			int leaf_idx = kd_tree_query_tree_sequential(
				patch,
				dists_local + i * n_neighbors,
				indices_local + i * n_neighbors,
				n_neighbors,
				max_leaves,
				1,
				kdtree_record
			);

			leaf_indices[i * (1 + n_neighbors) + 0] = leaf_idx;
		}


		// (2) propagate neighbors
		if (propagate > 0){
#pragma omp parallel for
			for (i=0; i < n_cols; i++){

				// get the nearest neighbor index of patch "above" (or "below" depending on how the image is stored, starting from the bottom or from the top)
				int location_above = n_cols * (patch_y - 1) + i;

				for (k=0; k<n_neighbors; k++){


					int best_idx_above = indices[location_above * n_neighbors + k];

					int prop_index_original = *(int*)(kdtree_record->XI + best_idx_above * (kdtree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(int)) + kdtree_record->dXtrain * sizeof(FLOAT_TYPE));

					// we use the index from the patch "above"; hence, a suitable
					// propagation candidate is to use the index + n_cols (one row below)
					if (prop_index_original + n_cols < npatchesred){
						prop_index_original = prop_index_original + n_cols;
					}

					// add leaf index to array
					int leaf_index = kdtree_record->points_leaf_indices[prop_index_original];
					leaf_indices[i * (1 + n_neighbors) + 1 + k] = leaf_index;

				}

			}
		}

		// do brute force using all leaf indices
#pragma omp parallel for
		for (i=0; i < n_cols; i++){

			brute_force_group(
					patches_row + i * dpatchesred,
					dpatchesred,
					leaf_indices + i * (1 + n_neighbors),
					n_neighbors,
					kdtree_record,
					params,
					indices_local + i * n_neighbors,
					dists_local + i * n_neighbors,
					propagate);

			// copy results to global nn_indices array (only indices are needed)
			for (k=0; k < n_neighbors; k++){
				indices[n_cols * patch_y * n_neighbors + i * n_neighbors + k] = indices_local[i * n_neighbors + k];
			}

		}

	}


	printf("Converting the indices (to fit with the original ordering) ...\n");
	char *XI = kdtree_record->XI;
	int size_elt = dpatchesred * sizeof(FLOAT_TYPE) + sizeof(int);

#pragma omp parallel for
	for (i = 0; i < npatchesred; i++) {
		for (j = 0; j < n_neighbors; j++) {

			int index_in_tree = indices[i * n_neighbors + j];
			indices[i * n_neighbors + j] = *((int *) (XI + index_in_tree * size_elt + dpatchesred * sizeof(FLOAT_TYPE)));

		}
	}


	if (select_best_nn > 0){

		// select first one
		printf("Finding the best nn indices (based on the original patches) ...\n");

#pragma omp parallel for
		for (i=0; i<npatchesred; i++){

			int best_idx=-1;
			int n;
			FLOAT_TYPE best_diff = MAX_FLOAT_TYPE;

			for(n=0; n<n_neighbors; n++){

				int current_idx = indices[i*n_neighbors + n];
				FLOAT_TYPE d = norm_array(patchesA + i*dpatchesA, patchesB + current_idx*dpatchesB, dpatchesA);
				if (d<best_diff){
					best_diff = d;
					best_idx = current_idx;
				}
			}

			finalindices[i] = best_idx;
		}
	} else {
		// select first one
		for (i=0; i<npatchesred; i++){
			finalindices[i] = indices[i*n_neighbors + 0];
		}
	}

    free(indices);
	free(indices_local);
	free(dists_local);
	free(leaf_indices);

}

FLOAT_TYPE norm_array(uint8_t *patchA, uint8_t *patchB, int dim){

	int i,j;
	FLOAT_TYPE d = 0.0;

	for (i=0; i<dim; i++){
		FLOAT_TYPE diff = (FLOAT_TYPE) patchA[i] - patchB[i];
		d += diff*diff;
	}
	d = sqrt(d);

	return d;
}
/**
 * Frees resources
 *
 */
void free_resources_extern(void) {

}
