/*
 * base.h
 *
 */

#ifndef INCLUDE_BASE_H_
#define INCLUDE_BASE_H_

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
//#include "float.h"

typedef struct futhark_ctx_inp {
	// unless otherwise specified by '_host', all arrays reside on GPU
	void*   fut_ctx;         // struct futhark_context*
	int tval;
	int supercharge;
    int k;
	int height;

	int platform_id;
    int device_id;
    int debug;

	int hImageA;
	int wImageA;
	int cImage;
	void*   imgA;  // struct futhark_f32_3d*, shape: [hImageA][wImageA][cImage]i32

	int hImageB;
	int wImageB;
	void*   imgB;  // struct futhark_f32_3d*, shape: [hImageB][wImageB][cImage]i32

	int psize;
	int patch_small;
	int patch_large;
	void*   comps; // struct futhark_f32_2d*, shape: [dim_reduced][psize*psize*cImage]
	void*   means; // struct futhark_f32_1d*, shape: [psize*psize*cImage]

	void*  nn_inds_host; // int32_t*, shape: [(hImageA-psize+1)*(wImageA-psize+1)]i32
	void*  nn_dsts_host; // float*  , shape: [(hImageA-psize+1)*(wImageA-psize+1)]f32
} FUTHARK_CTX_INP;

void fit_extern ( FUTHARK_CTX_INP *params, int debug );
void free_extern( FUTHARK_CTX_INP *params );
void pair_free( FUTHARK_CTX_INP *params );

void init_extern(
		FUTHARK_CTX_INP *params, // output
        int dim_reduced
	);

void pair_init(
		FUTHARK_CTX_INP *params, // output
		int *imageA, // input starts
		int himageA, // 800
		int wimageA, // 1920
		int cimageA, // 3
        int *imageB,
		int himageB, // 800
		int wimageB, // 1920
		int cimageB, // 3
		// float *components,
		// int n_components, // 16
		// int d_components, // 192
		// float *means,
		// int n_means, // 192
		int *indices, // result
		int n_indices,
		float *distances, // result
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
	);

#endif /* INCLUDE_BASE_H_ */
