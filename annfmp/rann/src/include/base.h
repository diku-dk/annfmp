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
	int32_t tval;
	int32_t supercharge;
    int64_t k;
	int32_t height;

    int32_t n_refer;
    int32_t n_query;
    int32_t dim;

    void* refer_pts;        // struct futhark_f32_2d*, shape [n_refer][dim]
    void* query_pts;        // struct futhark_f32_2d*, shape [n_query][dim]

    int32_t* nn_inds_host;  // shape [n_query * k]
} FUTHARK_CTX_INP;

void fit_extern ( FUTHARK_CTX_INP *params, int debug );
void free_extern( FUTHARK_CTX_INP *params );
void pair_free( FUTHARK_CTX_INP *params );

void init_extern(
		FUTHARK_CTX_INP *params, // output
        int dim
	);

void pair_init(
		FUTHARK_CTX_INP *params, // output
		float* refer_pts,
		int n_refer,
		int d_refer,

		float* query_pts,
		int n_query,
		int d_query,

		int* indices,
		int n_query_indices,

		int n_neighbors,
		int height,
		int tval,
		int supercharge
	);

#endif /* INCLUDE_BASE_H_ */
