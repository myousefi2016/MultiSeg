#ifndef CUDA_CUDA_TYPES_HPP
#define CUDA_CUDA_TYPES_HPP

#include "Config.hpp"

typedef int                  CudaSourceElement;
typedef LEVEL_SET_FIELD_TYPE CudaLevelSetElement;
typedef unsigned char        CudaTagElement;
typedef unsigned int         CudaCompactElement;
typedef uint4                CudaCompactElement4;

struct CudaConstraintValues
{
    float leftForegroundValues [ MAX_NUM_CONSTRAINT_VALUES ];
    float rightForegroundValues[ MAX_NUM_CONSTRAINT_VALUES ];
    float leftBackgroundValues [ MAX_NUM_CONSTRAINT_VALUES ];
    float rightBackgroundValues[ MAX_NUM_CONSTRAINT_VALUES ];

    int   numForegroundValues;
    int   numBackgroundValues;
};

#endif