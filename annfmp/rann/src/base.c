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
        int dim
) {
	struct futhark_context_config* fut_ctx_conf = futhark_context_config_new();
	int tile_size = (dim <= 32) ? dim : 16;
	futhark_context_config_set_default_tile_size(fut_ctx_conf, tile_size);
	futhark_context_config_set_profiling(fut_ctx_conf, 0);

	params->fut_ctx = futhark_context_new(fut_ctx_conf);
}

void pair_init(
		FUTHARK_CTX_INP* params, // output

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
) {
    if (d_refer != d_query) {
        fprintf(stderr, "RANN error: reference dim %d != query dim %d\n",
                d_refer, d_query);
        exit(1);
    }

    if (n_query_indices != n_query) {
        fprintf(stderr,
                "RANN error: index rows %d != query rows %d\n",
                n_query_indices,
                n_query);
        exit(1);
    }

	params->tval = tval;
    params->supercharge = supercharge;
    params->k = (int64_t)n_neighbors;
    params->height = height;

    params->n_refer = n_refer;
    params->n_query = n_query;
    params->dim = d_refer;

    params->nn_inds_host = indices;

	params->refer_pts = futhark_new_f32_2d(
        params->fut_ctx,
        refer_pts,
        n_refer,
        d_refer
    );

    params->query_pts = futhark_new_f32_2d(
        params->fut_ctx,
        query_pts,
        n_query,
        d_query
    );

	if (params->refer_pts == NULL || params->query_pts == NULL) {
        fprintf(stderr, "RANN error while creating Futhark input arrays: %s\n",
                futhark_context_get_error(params->fut_ctx));
        exit(1);
    }
}

void pair_free( FUTHARK_CTX_INP *params ) {
	struct futhark_context* fut_ctx = (struct futhark_context*) params->fut_ctx;

	int s = 0;
	s += futhark_free_f32_2d(fut_ctx, (struct futhark_f32_2d*)params->refer_pts);
	s += futhark_free_f32_2d(fut_ctx, (struct futhark_f32_2d*)params->query_pts);

	if (s != 0) {
    	printf("In pair_free: %s\nEXITING!\n", futhark_context_get_error(params->fut_ctx));
      	exit(1);
    }

    params->refer_pts = NULL;
    params->query_pts = NULL;
}

void free_extern( FUTHARK_CTX_INP *params ) {
	// one part
	struct futhark_context* fut_ctx = (struct futhark_context*) params->fut_ctx;

	// second part (called only once)
	futhark_context_clear_caches(fut_ctx);
	futhark_context_free(fut_ctx);

	params->fut_ctx = NULL;
}

/**
 * Fit extern
 *
 */
void fit_extern( FUTHARK_CTX_INP *params, int profile ) {

	// augmenting the parameters with the right types
	struct futhark_context* fut_ctx = (struct futhark_context*)params->fut_ctx;
	struct futhark_f32_2d* refer_pts = (struct futhark_f32_2d*)params->refer_pts;
	struct futhark_f32_2d* query_pts = (struct futhark_f32_2d*)params->query_pts;

	struct futhark_i32_2d* knn_inds = NULL;

   	int s = 0;

    if(profile) {
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
	if (s != 0) {
        fprintf(stderr,
                "RANN Futhark error: %s\n",
                futhark_context_get_error(fut_ctx));
        exit(1);
    }

    s = futhark_values_i32_2d(
        fut_ctx,
        knn_inds,
        params->nn_inds_host
    );

    if (s != 0) {
        fprintf(stderr,
                "RANN error copying indices back to host: %s\n",
                futhark_context_get_error(fut_ctx));
        exit(1);
    }

    s = futhark_free_i32_2d(
        fut_ctx,
        knn_inds
    );

    if (s != 0) {
        fprintf(stderr,
                "RANN error freeing result array: %s\n",
                futhark_context_get_error(fut_ctx));
        exit(1);
    }

    if (profile) {
        printf("Futhark report: %s", futhark_context_report(fut_ctx));
    }
}
