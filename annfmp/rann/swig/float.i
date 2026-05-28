%module wrapper_float

%{
#define SWIG_FILE_WITH_INIT
#include "../src/include/base.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(int* imageA, int himageA, int wimageA, int cimageA)}
%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(int* imageB, int himageB, int wimageB, int cimageB)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* components, int n_components, int d_components)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* means, int n_means)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* indices, int n_indices)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* distances, int n_distances)}

%include "../src/include/base.h"