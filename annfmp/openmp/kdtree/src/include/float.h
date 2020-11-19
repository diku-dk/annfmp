/*
 * float.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */
 
#ifndef INCLUDE_FLOAT_H_
#define INCLUDE_FLOAT_H_
 
#include <float.h>

#define DBL_EPSILON 2.2204460492503131E-16

#if USE_DOUBLE > 0
#define FLOAT_TYPE double
#define PARSE_FLOAT strtod
#define MAX_FLOAT_TYPE     1.7976931348623158e+308
#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#else
#define FLOAT_TYPE float
#define PARSE_FLOAT strtof
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38
#endif

#endif /* INCLUDE_FLOAT_H_ */
