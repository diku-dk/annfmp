%module wrapper_float

%{
#define SWIG_FILE_WITH_INIT
#include "../src/include/base.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

/* refer_pts: numpy float32 2D array -> float*, n_refer, d_refer */
%apply (float* IN_ARRAY2, int DIM1, int DIM2)
{
    (float* refer_pts, int n_refer, int d_refer)
};

/* query_pts: numpy float32 2D array -> float*, n_query, d_query */
%apply (float* IN_ARRAY2, int DIM1, int DIM2)
{
    (float* query_pts, int n_query, int d_query)
};

/* indices: numpy int32 2D array, modified in-place -> int*, n_query_indices, n_neighbors */
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2)
{
    (int* indices, int n_query_indices, int n_neighbors)
};

%include "../src/include/base.h"