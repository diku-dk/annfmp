#pragma once

// Headers

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>


// Initialisation

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_nvrtc_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_ptx_to(struct futhark_context_config *cfg,
                                        const char *path);
void futhark_context_config_load_ptx_from(struct futhark_context_config *cfg,
                                          const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);

// Arrays

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
CUdeviceptr futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                      struct futhark_f32_1d *arr);
const int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                                    struct futhark_f32_1d *arr);
struct futhark_f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
CUdeviceptr futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                      struct futhark_f32_2d *arr);
const int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                                    struct futhark_f32_2d *arr);
struct futhark_f32_3d ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2);
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2);
int futhark_free_f32_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d *arr);
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data);
CUdeviceptr futhark_values_raw_f32_3d(struct futhark_context *ctx,
                                      struct futhark_f32_3d *arr);
const int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                                    struct futhark_f32_3d *arr);
struct futhark_i32_1d ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx, const
                                          int32_t *data, int64_t dim0);
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0);
int futhark_free_i32_1d(struct futhark_context *ctx,
                        struct futhark_i32_1d *arr);
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data);
CUdeviceptr futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                      struct futhark_i32_1d *arr);
const int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                                    struct futhark_i32_1d *arr);
struct futhark_i32_2d ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx, const
                                          int32_t *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_i32_2d(struct futhark_context *ctx,
                        struct futhark_i32_2d *arr);
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data);
CUdeviceptr futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                      struct futhark_i32_2d *arr);
const int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                                    struct futhark_i32_2d *arr);
struct futhark_i32_3d ;
struct futhark_i32_3d *futhark_new_i32_3d(struct futhark_context *ctx, const
                                          int32_t *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2);
struct futhark_i32_3d *futhark_new_raw_i32_3d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2);
int futhark_free_i32_3d(struct futhark_context *ctx,
                        struct futhark_i32_3d *arr);
int futhark_values_i32_3d(struct futhark_context *ctx,
                          struct futhark_i32_3d *arr, int32_t *data);
CUdeviceptr futhark_values_raw_i32_3d(struct futhark_context *ctx,
                                      struct futhark_i32_3d *arr);
const int64_t *futhark_shape_i32_3d(struct futhark_context *ctx,
                                    struct futhark_i32_3d *arr);
struct futhark_u8_2d ;
struct futhark_u8_2d *futhark_new_u8_2d(struct futhark_context *ctx, const
                                        uint8_t *data, int64_t dim0,
                                        int64_t dim1);
struct futhark_u8_2d *futhark_new_raw_u8_2d(struct futhark_context *ctx, const
                                            CUdeviceptr data, int offset,
                                            int64_t dim0, int64_t dim1);
int futhark_free_u8_2d(struct futhark_context *ctx, struct futhark_u8_2d *arr);
int futhark_values_u8_2d(struct futhark_context *ctx, struct futhark_u8_2d *arr,
                         uint8_t *data);
CUdeviceptr futhark_values_raw_u8_2d(struct futhark_context *ctx,
                                     struct futhark_u8_2d *arr);
const int64_t *futhark_shape_u8_2d(struct futhark_context *ctx,
                                   struct futhark_u8_2d *arr);

// Opaque values


// Entry points

int futhark_entry_buildKDtree(struct futhark_context *ctx, int32_t *out0,
                              int32_t *out1, int32_t *out2,
                              struct futhark_f32_2d **out3,
                              struct futhark_i32_1d **out4,
                              struct futhark_i32_1d **out5,
                              struct futhark_i32_1d **out6,
                              struct futhark_f32_1d **out7,
                              struct futhark_i32_1d **out8, const int32_t in0,
                              const struct futhark_f32_2d *in1);
int futhark_entry_exactKnnFixK(struct futhark_context *ctx,
                               struct futhark_i32_2d **out0,
                               struct futhark_f32_2d **out1, int32_t *out2,
                               const struct futhark_f32_2d *in0, const
                               struct futhark_i32_1d *in1, const
                               struct futhark_f32_1d *in2, const
                               struct futhark_i32_1d *in3, const int32_t in4,
                               const struct futhark_f32_2d *in5, const
                               struct futhark_i32_1d *in6, const
                               struct futhark_i32_2d *in7, const
                               struct futhark_f32_2d *in8);
int futhark_entry_findNaturalLeavesFixK(struct futhark_context *ctx,
                                        struct futhark_i32_2d **out0,
                                        struct futhark_f32_2d **out1,
                                        struct futhark_i32_1d **out2, const
                                        struct futhark_f32_2d *in0, const
                                        struct futhark_i32_1d *in1, const
                                        struct futhark_f32_1d *in2, const
                                        struct futhark_f32_2d *in3);
int futhark_entry_mkImgPatches(struct futhark_context *ctx,
                               struct futhark_u8_2d **out0, const int32_t in0,
                               const struct futhark_i32_3d *in1);
int futhark_entry_propagateFixK(struct futhark_context *ctx,
                                struct futhark_i32_2d **out0,
                                struct futhark_f32_2d **out1, const int32_t in0,
                                const int32_t in1, const
                                struct futhark_f32_2d *in2, const
                                struct futhark_i32_1d *in3, const
                                struct futhark_i32_1d *in4, const
                                struct futhark_f32_2d *in5, const
                                struct futhark_i32_1d *in6, const
                                struct futhark_i32_2d *in7, const
                                struct futhark_f32_2d *in8);
int futhark_entry_reducePatchDim(struct futhark_context *ctx,
                                 struct futhark_f32_2d **out0, const
                                 struct futhark_u8_2d *in0, const
                                 struct futhark_f32_2d *in1, const
                                 struct futhark_f32_1d *in2);
int futhark_entry_selectBestNN(struct futhark_context *ctx,
                               struct futhark_i32_1d **out0,
                               struct futhark_f32_1d **out1, float *out2, const
                               int32_t in0, const struct futhark_i32_2d *in1,
                               const struct futhark_i32_3d *in2, const
                               struct futhark_i32_3d *in3);
int futhark_entry_selectBestNN_BAD(struct futhark_context *ctx,
                                   struct futhark_i32_1d **out0,
                                   struct futhark_f32_1d **out1, const
                                   int32_t in0, const
                                   struct futhark_i32_2d *in1, const
                                   struct futhark_f32_3d *in2, const
                                   struct futhark_f32_3d *in3);

// Miscellaneous

int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
#define FUTHARK_BACKEND_cuda
