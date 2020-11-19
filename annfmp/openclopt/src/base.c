/*
 * base.c
 */
#include "futhark/driverKNN.c"
#include "include/base.h"

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

void printKNNs(int32_t kk, int32_t n, int32_t* knn_indices, float* knn_distances) {
	printf("\nPriniting KNNs\n");
	for(int32_t i=0; i<n; i++) {
		printf("Patch %d: [(%d,%f)", i, knn_indices[i*kk], knn_distances[i*kk]);
		for(int32_t j=1; j<kk; j++) {
			printf(", (%d,%f)", knn_indices[i*kk+j], knn_distances[i*kk+j]);
		}
		printf("]\n");
	}
	printf("\n");
}

void print1Dfloat(int32_t n, float* arr) {
	printf("[%f", arr[0]);
	for(int32_t i=1; i<n; i++) {
		printf(", %f", arr[i]);
	}
	printf("]\n");
}

void print1Dint(int32_t n, int32_t* arr) {
	printf("[%d", arr[0]);
	for(int32_t i=1; i<n; i++) {
		printf(", %d", arr[i]);
	}
	printf("]\n");
}

void print2Dfloat(int32_t kk, int32_t n, float* arr) {
	printf("\nPriniting Float 2D array: [\n");
	for(int32_t i=0; i<n; i++) {
		printf("[%f", arr[i*kk]);
		for(int32_t j=1; j<kk; j++) {
			printf(", %f", arr[i*kk+j]);
		}
		printf("]\n");
	}
	printf("]\n");
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

//int main() { return 0; }

/**
 * Initializes the futhark context and the input buffers.
 */
void init_extern(
		FUTHARK_CTX_INP* params, // output
        int dim_reduced
) {

	// only call once
	struct futhark_context_config* fut_ctx_conf = futhark_context_config_new();
	int tile_size = (dim_reduced <= 32) ? dim_reduced : 16;
	futhark_context_config_set_default_tile_size(fut_ctx_conf, tile_size);
	futhark_context_config_set_profiling(fut_ctx_conf, 0);
	int s = 0;

	// thresholds for reducePatchDim
	const char* threshold_reddim_outer = "reducePatchDim.suff_outer_par_0";
	const char* threshold_reddim_inner = "reducePatchDim.suff_outer_par_1"; //(threshold (!reducePatchDim.suff_outer_par_0))
	s += futhark_context_config_set_size(fut_ctx_conf, threshold_reddim_outer, 100000000);
	s += futhark_context_config_set_size(fut_ctx_conf, threshold_reddim_inner, 4);

	// thresholds for buildKDtree
	const char* buildKDT_intra4 = "buildKDtree.suff_intra_par_4"; // (threshold (!buildKDtree.suff_outer_par_3))
	const char* buildKDT_intra7 = "buildKDtree.suff_intra_par_7"; // (threshold (!buildKDtree.suff_outer_par_3 !buildKDtree.suff_intra_par_4))
	const char* buildKDT_outer0 = "buildKDtree.suff_outer_par_0"; // (threshold ())
	const char* buildKDT_outer1 = "buildKDtree.suff_outer_par_1"; // (threshold ())
	const char* buildKDT_outer2 = "buildKDtree.suff_outer_par_2"; // (threshold ())
	const char* buildKDT_outer3 = "buildKDtree.suff_outer_par_3"; // (threshold ())

	s += futhark_context_config_set_size(fut_ctx_conf, buildKDT_intra4, 3000000);
	s += futhark_context_config_set_size(fut_ctx_conf, buildKDT_intra7, 3000000);
	s += futhark_context_config_set_size(fut_ctx_conf, buildKDT_outer0, 100000000);
	s += futhark_context_config_set_size(fut_ctx_conf, buildKDT_outer1, 100000000);
	s += futhark_context_config_set_size(fut_ctx_conf, buildKDT_outer2, 256);
	s += futhark_context_config_set_size(fut_ctx_conf, buildKDT_outer3, 100000000);

	// thresholds for exactKnnFixK
	const char* exactKNN_intra1 = "exactKnnFixK.suff_intra_par_1"; // (threshold (!exactKnnFixK.suff_outer_par_0))
	const char* exactKNN_outer0 = "exactKnnFixK.suff_outer_par_0"; // (threshold ())

	s += futhark_context_config_set_size(fut_ctx_conf, exactKNN_intra1, 16); //2048);         // 16
	s += futhark_context_config_set_size(fut_ctx_conf, exactKNN_outer0, 100000000); //256);  // 100000000

//	s += futhark_context_config_set_size(fut_ctx_conf, exactKNN_intra1, 2048);         // 16
//	s += futhark_context_config_set_size(fut_ctx_conf, exactKNN_outer0, 256);  // 100000000


	// thresholds for propagateFixK
	const char* threshold_propagate_intra= "propagateFixK.suff_intra_par_3";
	const char* threshold_propagate_outer= "propagateFixK.suff_outer_par_2";
	s += futhark_context_config_set_size(fut_ctx_conf, threshold_propagate_outer, 100000000); //256);  // 100000000
	s += futhark_context_config_set_size(fut_ctx_conf, threshold_propagate_intra, 16); //2048);   // 16

//	s += futhark_context_config_set_size(fut_ctx_conf, threshold_propagate_outer, 256);  // 100000000
//	s += futhark_context_config_set_size(fut_ctx_conf, threshold_propagate_intra, 2048);   // 16



	// thresholds for selectBestNN
	//const char* threshold_selectbest_intra = "selectBestNN.suff_intra_par_1";
	//const char* threshold_selectbest_outer = "selectBestNN.suff_outer_par_0";
	//s += futhark_context_config_set_size(fut_ctx_conf, threshold_selectbest_outer, 100000000);
	//s += futhark_context_config_set_size(fut_ctx_conf, threshold_selectbest_intra, 16);
	//selectBestNN.suff_outer_par_2 (threshold (!selectBestNN.suff_outer_par_0 !selectBestNN.suff_intra_par_1))

	// Cosmin: fixed the bug by hand by modifying by hand the cuda generated code;
	//		   this is a nasty bug, it will probably take a while to fix in Futhark.
	const char* filename = "../annfmp/openclopt/src/futhark/kernels.cu";
	//futhark_context_config_dump_program_to(fut_ctx_conf, filename);
	futhark_context_config_load_program_from(fut_ctx_conf, filename);

	// creating a new context must happen AFTER all the
	// context configuration parameters have been set!!!
	params->fut_ctx = futhark_context_new(fut_ctx_conf);

	// for each pair of images

}

void pair_init(
		FUTHARK_CTX_INP* params, // output
		int *imageA, // input starts
		int himageA, // 800
		int wimageA, // 1920
		int cimageA, // 3
        int *imageB,
		int himageB, // 800
		int wimageB, // 1920
		int cimageB, // 3
		float*  components,
		int n_components, // 16
		int d_components, // 192
		float*  means,
		int n_means, // 192
		int *indices, // result
		int n_indices,
		float* distances, // result
		int n_distances,
        int n_neighbors,
        int psize,
        int dim_reduced,
        int n_subset,
        int leaf_size,
        int seed,
		int platform_id,
        int device_id      // input ends
) {

#if 0
	printf("Height: %d, width: %d, c: %d\n", himageA, wimageA, cimageA);
	printf("Means: [%d]\n", n_means);
	print2Dfloat(1, n_means, means);
	printf("\nComponents [%d][%d]:\n", n_components, d_components);
	print2Dfloat(d_components, n_components, components);
#endif


	params->kk = 8;
	params->platform_id = platform_id;
    params->device_id = device_id;

    params->debug = 1;

    params->leaf_size = leaf_size;
	params->hImageA = himageA;
	params->wImageA = wimageA;
	params->cImage  = cimageA;
	params->imgA = imageA;  // shape: [hImageA][wImageA][cImage]i32

	params->hImageB = himageB;
	params->wImageB = wimageB;
	params->imgB = imageB;  // shape: [hImageB][wImageB][cImage]i32

	params->psize = psize;
	params->patch_small = dim_reduced;
	params->patch_large = params->psize * params->psize * params->cImage;
	params->comps = components; // shape: [psize*psize*cImage][patch_small]
	params->means = means;      // shape: [psize*psize*cImage]


	params->imgA = futhark_new_i32_3d(params->fut_ctx, imageA, himageA, wimageA, cimageA);
	params->imgB = futhark_new_i32_3d(params->fut_ctx, imageB, himageB, wimageB, cimageA);
	params->comps= futhark_new_f32_2d(params->fut_ctx, components, params->patch_small, params->patch_large);
	params->means= futhark_new_f32_1d(params->fut_ctx, means, params->patch_large);

	params->nn_inds_host = indices;   // shape: [(wimageA - psize + 1) * (himageA - psize + 1)]
	params->nn_dsts_host =  // shape: [(wimageA - psize + 1) * (himageA - psize + 1)]
		(float*)malloc((wimageA - psize + 1) * (himageA - psize + 1) * sizeof(float));

}

void pair_free( FUTHARK_CTX_INP *params ) {
	struct futhark_context_config* fut_ctx = (struct futhark_context_config*) params->fut_ctx;
	struct futhark_i32_3d* imgA   = (struct futhark_i32_3d*)params->imgA;
	struct futhark_i32_3d* imgB   = (struct futhark_i32_3d*)params->imgB;
	struct futhark_f32_2d* comps  = (struct futhark_f32_2d*)params->comps; 
	struct futhark_f32_1d* means  = (struct futhark_f32_1d*)params->means;

	int s = 0;
	s += futhark_free_f32_1d(fut_ctx, means);
	s += futhark_free_f32_2d(fut_ctx, comps);
	s += futhark_free_i32_3d(fut_ctx, imgA);
	s += futhark_free_i32_3d(fut_ctx, imgB);

	if (s != 0) {
    	printf("In free_extern: %s\nEXITING!\n", futhark_context_get_error(params->fut_ctx));
      	exit(1);
    }
	free(params->nn_dsts_host);


}
void free_extern( FUTHARK_CTX_INP *params ) {
	// one part
	struct futhark_context_config* fut_ctx = (struct futhark_context_config*) params->fut_ctx;

	// second part (called only once)
	futhark_context_clear_caches(fut_ctx);
	futhark_context_free(fut_ctx);
}

/**
 * Fit extern
 *
 */
void fit_extern( FUTHARK_CTX_INP *params, int profile ) {

	// augmenting the parameters with the right types
	struct futhark_context_config* fut_ctx = (struct futhark_context_config*) params->fut_ctx;
	struct futhark_i32_3d* imgA   = (struct futhark_i32_3d*)params->imgA;
	struct futhark_i32_3d* imgB   = (struct futhark_i32_3d*)params->imgB;
	struct futhark_f32_2d* comps  = (struct futhark_f32_2d*)params->comps; 
	struct futhark_f32_1d* means  = (struct futhark_f32_1d*)params->means;
	int32_t* nn_inds_host = (int32_t*)params->nn_inds_host;
	float*   nn_dsts_host = (float*  )params->nn_dsts_host;
	const int32_t psize = params->psize;
	const int32_t patch_small = params->patch_small;
	const int32_t patch_large = params->patch_large;
	const int32_t wImageA = params->wImageA;
	const int32_t hImageA = params->hImageA;
	const int32_t wImageB = params->wImageB;
	const int32_t hImageB = params->hImageB;
	const int32_t cImage  = params->cImage;
	const int32_t leaf_size = params->leaf_size;
	const int32_t kk      = params->kk;

	// creates patches
	int n_cols = wImageA - psize + 1;
	int n_rows = hImageA - psize + 1;

	if(profile)
		printf("Number of (rows-cols): (%d,%d), (patch_small, patch_large): (%d, %d), kk: %d, leaf-size:%d\n\n"
			  , n_rows, n_cols, patch_small, patch_large, kk, leaf_size);

	struct futhark_f32_2d* query_pts; //patches_A_reduced;
	struct futhark_f32_2d* refer_pts; //patches_B_reduced;

#if 1
	struct futhark_u8_2d* patches_A;
	struct futhark_u8_2d* patches_B;
	// 1. Patchify the images and reduce dimensionality
	if(profile) {
		unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL); 

      	int s = 0;
		s += futhark_entry_mkImgPatches(fut_ctx, &patches_A, psize, imgA);
		s += futhark_entry_reducePatchDim( fut_ctx, &query_pts //output
										 , patches_A, comps, means // input
										 );
		s += futhark_free_u8_2d(fut_ctx, patches_A);

		s += futhark_entry_mkImgPatches(fut_ctx, &patches_B, psize, imgB);
		s += futhark_entry_reducePatchDim( fut_ctx, &refer_pts //output
										 , patches_B, comps, means // input
										 );
		s += futhark_free_u8_2d(fut_ctx, patches_B);
		//futhark_context_sync(fut_ctx);
		cuCtxSynchronize();

		gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
      	if(profile)
      		printf("Patchifying the images (Futhark-CUDA): %lu microsecs\n", elapsed);

      	if (s != 0) {
      		printf("Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
	} else {
		futhark_entry_mkImgPatches(fut_ctx, &patches_A, psize, imgA);
		futhark_entry_reducePatchDim( fut_ctx, &query_pts
										 , patches_A, comps, means // input
		  						    );
		futhark_free_u8_2d(fut_ctx, patches_A);

		futhark_entry_mkImgPatches(fut_ctx, &patches_B, psize, imgB);
		futhark_entry_reducePatchDim( fut_ctx, &refer_pts //output
										 , patches_B, comps, means // input
									);
		futhark_free_u8_2d(fut_ctx, patches_B);
	}
#else
	// 2. Reducing the patch dimensionality without manifasting the
	//		large-patch array, i.e., directly from imgA and imgB.
	//    This slows it down quite a bit!
	//    You will have to modify and recompile the futhark file for this to work!
	if(profile) {
		printf("COSMIN 1\n");
		unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL); 

		int s1 = futhark_entry_reducePatchDim( fut_ctx, &query_pts //output
									, imgA, comps, means // input
									);
		int s2 = futhark_entry_reducePatchDim( fut_ctx, &refer_pts //output
									, imgB, comps, means // input
									);
		cuCtxSynchronize();
		printf("COSMIN 2\n");
		
		gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
      	if(profile)
      		printf("Reducing dimensionality(Futhark-CUDA): %lu microsecs\n", elapsed);

      	if (s1 != 0 || s2 != 0) {
      		printf("Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
	} else {
		futhark_entry_reducePatchDim( fut_ctx, &query_pts //output
									, imgA, comps, means // input
									);
		futhark_entry_reducePatchDim( fut_ctx, &refer_pts //output
									, imgB, comps, means // input
									);
	}
#endif

	int32_t height, num_inner_nodes, m_prime;
	struct futhark_f32_2d* leaves;
    struct futhark_i32_1d* indir;
    struct futhark_i32_1d* orig2leaf;
    struct futhark_i32_1d* median_dims;
    struct futhark_f32_1d* median_vals;
    struct futhark_i32_1d* clanc_eqdim;
      
    if(profile) { // 1. build the k-d tree
  	    unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL); 

      	//height, num_inner_nodes, m_pad, leafs, indir, median_dims, median_vals, clanc_eqdim =
      	//		self.futobj_mktree.buildKDtree(256, array_patches_b_reduced)
    	int s1 = futhark_entry_buildKDtree( fut_ctx
    							 , &height, &num_inner_nodes, &m_prime, &leaves, &indir // output
    							 , &orig2leaf, &median_dims, &median_vals, &clanc_eqdim // output
    							 , leaf_size, refer_pts // input
    							 );
    	cuCtxSynchronize();
    	
    	gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
      	if(profile)
      		printf("K-D Tree construction  (Futhark-CUDA): %lu microsecs\n", elapsed);

      	if (s1 != 0) {
      		printf("Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    } else {
    	futhark_entry_buildKDtree( fut_ctx
    							 , &height, &num_inner_nodes, &m_prime, &leaves, &indir // output
    							 , &orig2leaf, &median_dims, &median_vals, &clanc_eqdim // output
    							 , leaf_size, refer_pts // input
    							 );
    	//printf("Default leaf size: %d, actual leaf size: %d\n", leaf_size, m_prime / (1<<(height+1)) );
    }

    { // free original reference patch array.
    	int s = 0;
    	s += futhark_free_f32_2d(fut_ctx, refer_pts);
    	if (s != 0) {
      		printf("After Free Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    }

    struct futhark_i32_2d* knn_ini_inds;
    struct futhark_f32_2d* knn_ini_dsts;
    struct futhark_i32_1d* nat_leaves;
    
    if(profile) { // 2. for all queries, find the leaf to which the query naturally belongs to
  	    unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL); 

    	//  knn_ini_inds, knn_ini_dsts, nat_leaves = 
    	//		self.futobj_knn.findNaturalLeavesFixK(leaves, median_dims, median_vals, array_patches_a_reduced)

    	int s1 = futhark_entry_findNaturalLeavesFixK(fut_ctx,
    			&knn_ini_inds, &knn_ini_dsts, &nat_leaves, // output
            	leaves, median_dims, median_vals, query_pts// input
            );
    	cuCtxSynchronize();
    	
    	gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
      	if(profile)
	      	printf("Finding Natural Leaves (Futhark-CUDA): %lu microsecs\n", elapsed);

      	if (s1 != 0) {
      		printf("Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    } else {
    	futhark_entry_findNaturalLeavesFixK(fut_ctx,
    			&knn_ini_inds, &knn_ini_dsts, &nat_leaves, // output
            	leaves, median_dims, median_vals, query_pts// input
            );
    }

    struct futhark_i32_2d* knn_inds_exact;
    struct futhark_f32_2d* knn_dsts_exact;
    int32_t loop_count;

    if(profile) { // 3. compute the exact K-NNs for the first row
    	unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL); 

    	//  knn_inds_exact, knn_dsts_exact, loop_count = 
    	//	self.futobj_knn.exactKnnFixK(leafs, median_dims, median_vals, clanc_eqdim, self._n_cols, array_patches_a_reduced, nat_leaves, knn_ini_inds, knn_ini_dsts)
    	int s1 = futhark_entry_exactKnnFixK(fut_ctx,
    			&knn_inds_exact, &knn_dsts_exact, &loop_count, // output
    			leaves, median_dims, median_vals, clanc_eqdim, n_cols, // input
                query_pts, nat_leaves, knn_ini_inds, knn_ini_dsts // input
            );
    	cuCtxSynchronize();

    	gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
      	if(profile)
	      	printf("Exact k-NN Computation (Futhark-CUDA): %lu microsecs, num-iter: %d\n", elapsed, loop_count);

      	if (s1 != 0) {
      		printf("Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    } else {
    	futhark_entry_exactKnnFixK(fut_ctx,
    			&knn_inds_exact, &knn_dsts_exact, &loop_count, // output
    			leaves, median_dims, median_vals, clanc_eqdim, n_cols, // input
                query_pts, nat_leaves, knn_ini_inds, knn_ini_dsts // input
            );
    }

    { // we can now free some of the k-d tree structures, and the ini_inds/dsts
    	int s = 0;
    	s += futhark_free_i32_1d(fut_ctx, median_dims);
		s += futhark_free_f32_1d(fut_ctx, median_vals);
		s += futhark_free_i32_1d(fut_ctx, clanc_eqdim);
		s += futhark_free_i32_2d(fut_ctx, knn_ini_inds);
		s += futhark_free_f32_2d(fut_ctx, knn_ini_dsts);

		if (s != 0) {
      		printf("After Free Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    }

    struct futhark_i32_2d* knn_inds_all;
    struct futhark_f32_2d* knn_dsts_all;

    if (profile) { // 4. final propagation step
    	unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL); 

    	// knn_inds, knn_dsts = self.futobj_knn.propagateFixK(height, self._n_rows, leafs, indir, array_patches_a_reduced, nat_leaves, knn_inds_exact, knn_dsts_exact)
    	int s1 = futhark_entry_propagateFixK(fut_ctx,
                &knn_inds_all, &knn_dsts_all, // output
                height, n_rows, leaves, indir, orig2leaf, query_pts, // input
                nat_leaves, knn_inds_exact, knn_dsts_exact           // input
            );
    	cuCtxSynchronize();

    	gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
      	if(profile)
	      	printf("Final Propagation Step (Futhark-CUDA): %lu microsecs\n", elapsed);

      	if (s1 != 0) {
      		printf("Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    } else {
    	futhark_entry_propagateFixK(fut_ctx,
                &knn_inds_all, &knn_dsts_all, // output
                height, n_rows, leaves, indir, orig2leaf, query_pts, // input
                nat_leaves, knn_inds_exact, knn_dsts_exact           // input
            );
    }

#if 0
    {
	    int32_t* knn_indices = (int32_t*)malloc(n_cols * n_rows * kk * sizeof(int32_t));
    	float* knn_distances = (float*)  malloc(n_cols * n_rows * kk * sizeof(float  ));

		int s = 0;
		s += futhark_values_i32_2d(fut_ctx, knn_inds_all, knn_indices);
		s += futhark_values_f32_2d(fut_ctx, knn_dsts_all, knn_distances);
		if (s != 0) {
      		printf("GetArrayValue Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
      	if(profile)
      		printKNNs(kk, 32, knn_indices + 599*n_cols*kk, knn_distances + 599*n_cols*kk);
      	
	    free(knn_indices);
    	free(knn_distances);
    }
#endif

    { // free many buffers
    	int s = 0;
    	s += futhark_free_f32_2d(fut_ctx, query_pts);
		s += futhark_free_i32_2d(fut_ctx, knn_inds_exact);
		s += futhark_free_f32_2d(fut_ctx, knn_dsts_exact);
		s += futhark_free_f32_2d(fut_ctx, leaves);
		s += futhark_free_i32_1d(fut_ctx, indir);
		s += futhark_free_i32_1d(fut_ctx, orig2leaf);
		s += futhark_free_i32_1d(fut_ctx, nat_leaves);
		s += futhark_free_f32_2d(fut_ctx, knn_dsts_all);
		
		//s += futhark_free_i32_2d(fut_ctx, knn_inds_all);
		
		if (s != 0) {
      		printf("After Free Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    }

    struct futhark_i32_1d* nn_inds;
    struct futhark_f32_1d* nn_dsts;
    float  error = 33333333.3333;
    
    if(profile) { // finally selecting the best nearest neigbor from the original image
		unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL); 

    	int s = futhark_entry_selectBestNN(
    							fut_ctx, &nn_inds, &nn_dsts, &error, // output
                               	psize, knn_inds_all, imgA, imgB      // input
                            );
    	cuCtxSynchronize();

    	gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
      	if(profile)
	      	printf("Selecting Best Neighbor(Futhark-CUDA): %lu microsecs, error: %f\n", elapsed, error);

      	if (s != 0) {
      		printf("Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    } else {
    	futhark_entry_selectBestNN(
    							fut_ctx, &nn_inds, &nn_dsts, &error, // output
                               	psize, knn_inds_all, imgA, imgB      // input
                            );
    }

    if(profile)
    	printf("Futhark report: %s", futhark_context_report(fut_ctx));

    {
		int s = 0;
		s += futhark_values_i32_1d(fut_ctx, nn_inds, nn_inds_host);
		s += futhark_values_f32_1d(fut_ctx, nn_dsts, nn_dsts_host);

		if (s != 0) {
      		printf("GetArrayValue Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
#if 0
      	printf("\nBest NN for the first 16 queries:\n");
      	print1Dint  (16, nn_inds_host + 599*n_cols);
      	print1Dfloat(16, nn_dsts_host + 599*n_cols);
#endif
    }

    { // Free cuda memory
    	int s1 = 0;
		
		s1 += futhark_free_i32_2d(fut_ctx, knn_inds_all);
		s1 += futhark_free_i32_1d(fut_ctx, nn_inds);
		s1 += futhark_free_f32_1d(fut_ctx, nn_dsts);

		//printf("Report: %s\n", futhark_context_report(fut_ctx));

		if (s1 != 0) {
      		printf("After free Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    }
}

// Python commands:
//source .venvs/annfmp/bin/activate
//deactivate
//python setup.py clean; python setup.py develop
