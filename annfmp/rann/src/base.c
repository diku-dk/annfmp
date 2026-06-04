/*
 * base.c
 */
#include "futhark/knn-driver.c"
#include "include/base.h"

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

//int main() { return 0; }

/**
 * Initializes the futhark context and the input buffers.
 */
void init_extern(
		FUTHARK_CTX_INP* params, // output
        int dim_reduced
) {
	struct futhark_context_config* fut_ctx_conf = futhark_context_config_new();
	int tile_size = (dim_reduced <= 32) ? dim_reduced : 16;
	futhark_context_config_set_default_tile_size(fut_ctx_conf, tile_size);
	futhark_context_config_set_profiling(fut_ctx_conf, 0);
    int s = 0;

	params->fut_ctx = futhark_context_new(fut_ctx_conf);
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
		// float*  components,
		// int n_components, // 16
		// int d_components, // 192
		// float*  means,
		// int n_means, // 192
		int *indices, // result
		int n_indices,
		float* distances, // result
		int n_distances,
        int n_neighbors,
        int psize,
        int dim_reduced,
        int n_subset,
        int seed,
		int platform_id,
        int device_id,  
        int height,
		int tval,
        int supercharge
) {

	params->tval = tval;
    params->supercharge = supercharge;
    params->k = (int64_t)n_neighbors;

	params->platform_id = platform_id;
    params->device_id = device_id;

    params->debug = 1;

	params->height = height;
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
	// params->comps = components; // shape: [psize*psize*cImage][patch_small]
	// params->means = means;      // shape: [psize*psize*cImage]


	params->imgA = futhark_new_i32_3d(params->fut_ctx, imageA, himageA, wimageA, cimageA);
	params->imgB = futhark_new_i32_3d(params->fut_ctx, imageB, himageB, wimageB, cimageA);
	// params->comps= futhark_new_f32_2d(params->fut_ctx, components, params->patch_small, params->patch_large);
	// params->means= futhark_new_f32_1d(params->fut_ctx, means, params->patch_large);

	params->nn_inds_host = indices;   // shape: [(wimageA - psize + 1) * (himageA - psize + 1)]
	params->nn_dsts_host =  // shape: [(wimageA - psize + 1) * (himageA - psize + 1)]
		(float*)malloc((wimageA - psize + 1) * (himageA - psize + 1) * sizeof(float));

}

void pair_free( FUTHARK_CTX_INP *params ) {
	struct futhark_context* fut_ctx = (struct futhark_context*) params->fut_ctx;

    struct futhark_i32_3d* imgA   = (struct futhark_i32_3d*)params->imgA;
	struct futhark_i32_3d* imgB   = (struct futhark_i32_3d*)params->imgB;
	// struct futhark_f32_2d* comps  = (struct futhark_f32_2d*)params->comps;
	// struct futhark_f32_1d* means  = (struct futhark_f32_1d*)params->means;

    int s = 0;
	// s += futhark_free_f32_1d(fut_ctx, means);
	// s += futhark_free_f32_2d(fut_ctx, comps);
	s += futhark_free_i32_3d(fut_ctx, imgA);
	s += futhark_free_i32_3d(fut_ctx, imgB);

	if (s != 0) {
    	printf("In pair_free: %s\nEXITING!\n", futhark_context_get_error(params->fut_ctx));
      	exit(1);
    }
    free(params->nn_dsts_host);
}

void free_extern( FUTHARK_CTX_INP *params ) {
	// one part
	struct futhark_context* fut_ctx = (struct futhark_context*) params->fut_ctx;

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
	// struct futhark_f32_2d* comps  = (struct futhark_f32_2d*)params->comps;
	// struct futhark_f32_1d* means  = (struct futhark_f32_1d*)params->means;
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
	//const int32_t leaf_size = params->leaf_size;
	const int32_t height = params->height;
	const int32_t k = params->k;

	// creates patches
	int n_cols = wImageA - psize + 1;
	int n_rows = hImageA - psize + 1;
    struct futhark_i32_2d* knn_inds = NULL;

	printf("Number of (rows-cols): (%d,%d), (patch_small, patch_large): (%d, %d), kk: %d, height:%d\n\n"
			  , n_rows, n_cols, patch_small, patch_large, k, height);

	struct futhark_f32_2d* query_pts; //patches_A_reduced;
	struct futhark_f32_2d* refer_pts; //patches_B_reduced;

    struct futhark_u8_2d* patches_A;
	struct futhark_u8_2d* patches_B;
	// 1. Patchify the images and reduce dimensionality
	if(profile) {
		unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL);

      	int s = 0;
		// Used to be called patches_A
		s += futhark_entry_mkImgPatches(fut_ctx, &patches_A, psize, imgA);
		s += futhark_entry_reducePatchDim( fut_ctx, &query_pts //output
										 , patches_A// input
										 );
		s += futhark_free_u8_2d(fut_ctx, patches_A);

		s += futhark_entry_mkImgPatches(fut_ctx, &patches_B, psize, imgB);
		s += futhark_entry_reducePatchDim( fut_ctx, &refer_pts //output
										 , patches_B // input
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
										 , patches_A // input
		  						    );
		futhark_free_u8_2d(fut_ctx, patches_A);

		futhark_entry_mkImgPatches(fut_ctx, &patches_B, psize, imgB);
		futhark_entry_reducePatchDim( fut_ctx, &refer_pts //output
										 , patches_B // input
									);
		futhark_free_u8_2d(fut_ctx, patches_B);
    }

    if(profile) {
        int s = 0;
		unsigned long int elapsed;
	    struct timeval t_start, t_end, t_diff;
      	gettimeofday(&t_start, NULL);

		if (params->supercharge == 1) {
            s+= futhark_entry_mainSuper(
            fut_ctx,
            &knn_inds,
            params->tval,
            params->k,
            params->height,
            refer_pts,
            query_pts
            );
        } else {
            s+= futhark_entry_main(
                fut_ctx,
                &knn_inds,
                params->tval,
                params->k,
                params->height,
                refer_pts,
                query_pts
            );
        }

		//futhark_context_sync(fut_ctx);
		cuCtxSynchronize();

		gettimeofday(&t_end, NULL);
      	timeval_subtract(&t_diff, &t_end, &t_start);
      	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

		printf("RANN knn computation: %lu microsecs\n", elapsed);

	} else {
        int s = 0;
		if (params->supercharge == 1) {
            s+= futhark_entry_mainSuper(
            fut_ctx,
            &knn_inds,
            params->tval,
            params->k,
            params->height,
            refer_pts,
            query_pts
            );
        } else {
            s+= futhark_entry_main(
                fut_ctx,
                &knn_inds,
                params->tval,
                params->k,
                params->height,
                refer_pts,
                query_pts
            );
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
                               	psize, knn_inds, imgA, imgB      // input
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
    	int s = futhark_entry_selectBestNN(
    							fut_ctx, &nn_inds, &nn_dsts, &error, // output
                               	psize, knn_inds, imgA, imgB      // input
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


    if (profile) {
        printf("Futhark report: %s", futhark_context_report(fut_ctx));
    }

    { // Free cuda memory
    	int s1 = 0;

		s1 += futhark_free_i32_1d(fut_ctx, nn_inds);
		s1 += futhark_free_f32_1d(fut_ctx, nn_dsts);

		//printf("Report: %s\n", futhark_context_report(fut_ctx));

		if (s1 != 0) {
      		printf("After free Error: %s\nEXITING!\n", futhark_context_get_error(fut_ctx));
      		exit(1);
      	}
    }

	
}
