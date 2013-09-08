#ifndef SRC_CUDA_UPDATE_LEVEL_SET_VOLUME_CU
#define SRC_CUDA_UPDATE_LEVEL_SET_VOLUME_CU

template< typename T >
__global__ void  UpdateLevelSetVolumeKernel( CudaLevelSetElement* deviceLevelSetData,
                                             CudaTagElement*      deviceLevelSetExportData,
                                             CudaTagElement*      deviceTimeDerivativeData,
                                             float*               deviceFeatureSpaceDistanceToForegroundVolume,
                                             float*               deviceFeatureSpaceDistanceToBackgroundVolume,                                             
                                             dim3                 volumeDimensions,
                                             size_t               numActiveVoxels,
                                             int                  target,
                                             int                  maxDistanceBeforeShrink,
                                             float                curvatureInfluence,
                                             float                timeStep );

template< typename T >
__device__ void  UpdateLevelSetVolumeKernelHelper( CudaCompactElement   packedVoxelCoordinate,
                                                   CudaLevelSetElement* deviceLevelSetData,
                                                   CudaTagElement*      deviceLevelSetExportData,
                                                   CudaTagElement*      deviceTimeDerivativeData,
                                                   float*               deviceFeatureSpaceDistanceToForegroundVolume,
                                                   float*               deviceFeatureSpaceDistanceToBackgroundVolume,                                                   
                                                   dim3                 volumeDimensions,
                                                   int                  target,
                                                   int                  maxDistanceBeforeShrink,
                                                   float                curvatureInfluence,
                                                   float                timeStep );

extern "C" void CudaUpdateLevelSetVolumeAsync( CudaLevelSetElement*            levelSetData,
                                               CudaTagElement*                 levelSetExportData,
                                               CudaTagElement*                 timeDerivativeData,
                                               thrust::device_vector< float >& featureSpaceDistanceToForegroundVolumeDevice,
                                               thrust::device_vector< float >& featureSpaceDistanceToBackgroundVolumeDevice,                                               
                                               size_t                          numActiveElementsHost,
                                               dim3                            volumeDimensions,
                                               int                             target,
                                               int                             maxDistanceBeforeShrink,
                                               float                           curvatureInfluence,
                                               float                           timeStep,
                                               unsigned int                    numBytesPerVoxel,
                                               bool                            isSigned )
{
#ifdef CUDA_ARCH_SM_10
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 256, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( numActiveElementsHost / ( 256.0f * 4.0f ) ) );
#endif

#ifdef CUDA_ARCH_SM_13
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 512, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( numActiveElementsHost / ( 512.0f * 4.0f ) ) );
#endif

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocks, 1, 1 );

    if ( numThreadBlocks > 0 )
    {
        // call our kernel
        if ( numBytesPerVoxel == 1 && !isSigned )
        {
            UpdateLevelSetVolumeKernel< unsigned char ><<< gridDimensions, threadBlockDimensions >>>(
                levelSetData,
                levelSetExportData,
                timeDerivativeData,
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToForegroundVolumeDevice[ 0 ] ),
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToBackgroundVolumeDevice[ 0 ] ),
                volumeDimensions,
                numActiveElementsHost,
                target,
                maxDistanceBeforeShrink,
                curvatureInfluence,
                timeStep );
        }
        else
        if ( numBytesPerVoxel == 1 && isSigned )
        {
            UpdateLevelSetVolumeKernel< char ><<< gridDimensions, threadBlockDimensions >>>(
                levelSetData,
                levelSetExportData,
                timeDerivativeData,
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToForegroundVolumeDevice[ 0 ] ),
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToBackgroundVolumeDevice[ 0 ] ),
                volumeDimensions,
                numActiveElementsHost,
                target,
                maxDistanceBeforeShrink,
                curvatureInfluence,
                timeStep );
        }
        else
        if ( numBytesPerVoxel == 2 && !isSigned )
        {
            UpdateLevelSetVolumeKernel< unsigned short ><<< gridDimensions, threadBlockDimensions >>>(
                levelSetData,
                levelSetExportData,
                timeDerivativeData,
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToForegroundVolumeDevice[ 0 ] ),
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToBackgroundVolumeDevice[ 0 ] ),
                volumeDimensions,
                numActiveElementsHost,
                target,
                maxDistanceBeforeShrink,
                curvatureInfluence,
                timeStep );
        }
        else
        if ( numBytesPerVoxel == 2 && isSigned )
        {
            UpdateLevelSetVolumeKernel< short ><<< gridDimensions, threadBlockDimensions >>>(
                levelSetData,
                levelSetExportData,
                timeDerivativeData,
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToForegroundVolumeDevice[ 0 ] ),
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToBackgroundVolumeDevice[ 0 ] ),
                volumeDimensions,
                numActiveElementsHost,
                target,
                maxDistanceBeforeShrink,
                curvatureInfluence,
                timeStep );
        }
        else
        if ( numBytesPerVoxel == 4 && !isSigned )
        {
            UpdateLevelSetVolumeKernel< unsigned int ><<< gridDimensions, threadBlockDimensions >>>(
                levelSetData,
                levelSetExportData,
                timeDerivativeData,
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToForegroundVolumeDevice[ 0 ] ),
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToBackgroundVolumeDevice[ 0 ] ),
                volumeDimensions,
                numActiveElementsHost,
                target,
                maxDistanceBeforeShrink,
                curvatureInfluence,
                timeStep );
        }
        else
        if ( numBytesPerVoxel == 4 && isSigned )
        {
            UpdateLevelSetVolumeKernel< int ><<< gridDimensions, threadBlockDimensions >>>(
                levelSetData,
                levelSetExportData,
                timeDerivativeData,
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToForegroundVolumeDevice[ 0 ] ),
                thrust::raw_pointer_cast< float >( &featureSpaceDistanceToBackgroundVolumeDevice[ 0 ] ),
                volumeDimensions,
                numActiveElementsHost,
                target,
                maxDistanceBeforeShrink,
                curvatureInfluence,
                timeStep );
        }
        else
        {
            Assert( 0 );
        }
    }
}

template< typename T >
__global__ void UpdateLevelSetVolumeKernel( CudaLevelSetElement* deviceLevelSetData,
                                            CudaTagElement*      deviceLevelSetExportData,
                                            CudaTagElement*      deviceTimeDerivativeData,
                                            float*               deviceFeatureSpaceDistanceToForegroundVolume,
                                            float*               deviceFeatureSpaceDistanceToBackgroundVolume,
                                            dim3                 volumeDimensions,
                                            size_t               numActiveVoxels,
                                            int                  target,
                                            int                  maxDistanceBeforeShrink,
                                            float                curvatureInfluence,
                                            float                timeStep )
{
    int arrayIndexInActiveElementTexture = ComputeIndexThread1DBlock1DTo1D();
    int arrayIndexInActiveElementList    = arrayIndexInActiveElementTexture * 4;

    if ( arrayIndexInActiveElementList < numActiveVoxels )
    {
        CudaCompactElement4 packedCoordinates = tex1Dfetch( CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D, arrayIndexInActiveElementTexture );

        UpdateLevelSetVolumeKernelHelper< T >( packedCoordinates.x,
                                               deviceLevelSetData,
                                               deviceLevelSetExportData,
                                               deviceTimeDerivativeData,
                                               deviceFeatureSpaceDistanceToForegroundVolume,
                                               deviceFeatureSpaceDistanceToBackgroundVolume,
                                               volumeDimensions,
                                               target,
                                               maxDistanceBeforeShrink,
                                               curvatureInfluence,
                                               timeStep );                
        arrayIndexInActiveElementList++;

        if ( arrayIndexInActiveElementList < numActiveVoxels )
        {
            UpdateLevelSetVolumeKernelHelper< T >( packedCoordinates.y,
                                                   deviceLevelSetData,
                                                   deviceLevelSetExportData,
                                                   deviceTimeDerivativeData,
                                                   deviceFeatureSpaceDistanceToForegroundVolume,
                                                   deviceFeatureSpaceDistanceToBackgroundVolume,
                                                   volumeDimensions,
                                                   target,
                                                   maxDistanceBeforeShrink,
                                                   curvatureInfluence,
                                                   timeStep );                
            arrayIndexInActiveElementList++;

            if ( arrayIndexInActiveElementList < numActiveVoxels )
            {
                UpdateLevelSetVolumeKernelHelper< T >( packedCoordinates.z,
                                                       deviceLevelSetData,
                                                       deviceLevelSetExportData,
                                                       deviceTimeDerivativeData,
                                                       deviceFeatureSpaceDistanceToForegroundVolume,
                                                       deviceFeatureSpaceDistanceToBackgroundVolume,
                                                       volumeDimensions,
                                                       target,
                                                       maxDistanceBeforeShrink,
                                                       curvatureInfluence,
                                                       timeStep );                
                arrayIndexInActiveElementList++;

                if ( arrayIndexInActiveElementList < numActiveVoxels )
                {
                    UpdateLevelSetVolumeKernelHelper< T >( packedCoordinates.w,
                                                           deviceLevelSetData,
                                                           deviceLevelSetExportData,
                                                           deviceTimeDerivativeData,
                                                           deviceFeatureSpaceDistanceToForegroundVolume,
                                                           deviceFeatureSpaceDistanceToBackgroundVolume,
                                                           volumeDimensions,
                                                           target,
                                                           maxDistanceBeforeShrink,
                                                           curvatureInfluence,
                                                           timeStep );                
                }
            }
        }
    }
}

__device__ float ComputeDensityTerm( float        featureSpaceDistanceToForeground,
                                     unsigned int maxDifferenceBeforeShrink )
{
    float f = ( -1.0f / maxDifferenceBeforeShrink ) * abs( featureSpaceDistanceToForeground ) + 1.0f;

    return f;
}

template< typename T >
__device__ void  UpdateLevelSetVolumeKernelHelper( CudaCompactElement   packedVoxelCoordinate,
                                                   CudaLevelSetElement* deviceLevelSetData,
                                                   CudaTagElement*      deviceLevelSetExportData,
                                                   CudaTagElement*      deviceTimeDerivativeData,
                                                   float*               deviceFeatureSpaceDistanceToForegroundVolume,
                                                   float*               deviceFeatureSpaceDistanceToBackgroundVolume,
                                                   dim3                 volumeDimensions,
                                                   int                  target,
                                                   int                  maxDistanceBeforeShrink,
                                                   float                curvatureInfluence,
                                                   float                timeStep )
{

    dim3 elementCoordinates = UnpackCoordinates( packedVoxelCoordinate );

    //
    // get 3x3x3 neighborhood
    //
    int3 currentCoordinates;
    dim3 currentCoordinatesDim3;
    int  arrayIndex;
    int  elementIndex;
    int  elementIndexUntiled;

    elementIndex        = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    elementIndexUntiled = ComputeIndex3DTo1D( elementCoordinates, volumeDimensions );

    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4negativeZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
                                                                         
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
                                                                         
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4positiveZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );

    GET_LEFT_SOURCE_NEIGHBORHOOD_HELPER ( u4LeftSource,  currentCoordinates, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0, T );
    GET_RIGHT_SOURCE_NEIGHBORHOOD_HELPER( u4RightSource, currentCoordinates, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0, T );

    float featureSpaceDistanceToForeground = deviceFeatureSpaceDistanceToForegroundVolume[ elementIndexUntiled ];
    float featureSpaceDistanceToBackground = deviceFeatureSpaceDistanceToForegroundVolume[ elementIndexUntiled ];
    
    // compute derivatives
    float derivativeCentralX          = ( u5 - u3 )                   / 2.0f;
    float derivativeCentralY          = ( u7 - u1 )                   / 2.0f;
    float derivativeCentralZ          = ( u4positiveZ - u4negativeZ ) / 2.0f; 

    float derivativeForwardX          = ( u5          - u4 );
    float derivativeForwardY          = ( u7          - u4 );
    float derivativeForwardZ          = ( u4positiveZ - u4 );

    float derivativeBackwardX         = ( u4 - u3 );
    float derivativeBackwardY         = ( u4 - u1 );
    float derivativeBackwardZ         = ( u4 - u4negativeZ );

    float secondDerivativeX = derivativeForwardX - derivativeBackwardX;
    float secondDerivativeY = derivativeForwardY - derivativeBackwardY;
    float secondDerivativeZ = derivativeForwardZ - derivativeBackwardZ;

    // compute speed function terms
    float connectivityTerm = 0.0f;

    if ( curvatureInfluence > 0.0f )
    {
        if ( Equals( u4positiveZ, u4negativeZ, 0.1f ) &&            
             Equals( u4negativeZ, u1,          0.1f ) &&
             Equals( u1, u3,                   0.1f ) &&
             Equals( u3, u5,                   0.1f ) &&
             Equals( u5, u7,                   0.1f ) &&
             Equals( u7, u4positiveZ,          0.1f ) )
        {
            float average    = ( u4positiveZ + u1 + u3 + u5 + u7 + u4negativeZ ) / 6.0f;
            connectivityTerm = average - u4;
        }
    }

    float curvatureTerm  = ( secondDerivativeX + secondDerivativeY + secondDerivativeZ );
    
    float densityTerm    = - ComputeDensityTerm( featureSpaceDistanceToForeground, maxDistanceBeforeShrink );
    
    if ( timeStep < 0 )
    {
        timeStep    = - timeStep;
        densityTerm = - densityTerm;
    }

    float speedFunction  = ( ( 1 - curvatureInfluence ) * ( densityTerm ) ) + ( ( curvatureInfluence ) * curvatureTerm );

    // gradient magnitude
    float gradientLength = sqrt( Sqr( derivativeCentralX ) + Sqr( derivativeCentralY ) + Sqr( derivativeCentralZ ) );

    // compute new value
    float levelSetDelta    = ( ( timeStep * speedFunction * gradientLength ) ) + connectivityTerm;
    float newLevelSetValue = min( 1.0f, max( -1.0f, u4 + levelSetDelta ) );

    // rescale
    newLevelSetValue = min( 1.0f, max( -1.0f, ( ( ( 1.0f + LEVEL_SET_RESCALE_AMOUNT ) * newLevelSetValue ) + LEVEL_SET_RESCALE_AMOUNT ) ) );

    // assign new value
#ifdef LEVEL_SET_FIELD_FIXED_POINT
    int writeValue                     = __float2int_rd( newLevelSetValue * LEVEL_SET_FIELD_MAX_VALUE );
    deviceLevelSetData[ elementIndex ] = writeValue;
#else
    deviceLevelSetData[ elementIndex ] = newLevelSetValue;
#endif

    int exportValue                                 = __float2int_rd( newLevelSetValue * LEVEL_SET_FIELD_EXPORT_MAX_VALUE );
    deviceLevelSetExportData[ elementIndexUntiled ] = exportValue;


    // compute and assign time derivative
    float timeDerivative = newLevelSetValue - u4;

    int timeDerivativeValue = abs( timeDerivative ) > TEMPORAL_DERIVATIVE_THRESHOLD ? 1 : 0;
    deviceTimeDerivativeData[ elementIndex ] = timeDerivativeValue;
}






struct Index1DToIndex3DFunction : public thrust::unary_function< int, int3 >
{
    int mNumVoxelsX, mNumVoxelsXY;
    float mInvNumVoxelsX, mInvNumVoxelsXY;

    Index1DToIndex3DFunction( int3 numVoxels ) :
        mNumVoxelsX    ( numVoxels.x ),
        mNumVoxelsXY   ( numVoxels.x * numVoxels.y ),
        mInvNumVoxelsX ( 1.0f / (float)mNumVoxelsX ),
        mInvNumVoxelsXY( 1.0f / (float)mNumVoxelsXY )
    {}

    __device__
    int3 operator() ( int index1D )
    {
        int z        = __float2uint_rd( index1D * mInvNumVoxelsXY );
        index1D      = index1D - ( z * mNumVoxelsXY );
        int y        = __float2uint_rd( index1D * mInvNumVoxelsX );
        int x        = index1D - ( y * mNumVoxelsX );

        int3 index3D = make_int3( x, y, z );

        return index3D;
    }
};

__constant__ CudaConstraintValues constantMemoryCudaConstraintValues;

struct InitializeFeatureSpaceDistanceMapsFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        FEATURE_SPACE_DISTANCE_TO_FOREGROUND,
        FEATURE_SPACE_DISTANCE_TO_BACKGROUND        
    };

    InitializeFeatureSpaceDistanceMapsFunction() {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3  index3D            = thrust::get< INDEX_3D >( tuple );
        int3  elementCoordinates = index3D;
        int3  currentCoordinates;
        
        GET_LEFT_SOURCE_NEIGHBORHOOD_HELPER ( u4LeftSource,  currentCoordinates, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0, unsigned char );
        GET_RIGHT_SOURCE_NEIGHBORHOOD_HELPER( u4RightSource, currentCoordinates, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0, unsigned char );

        float2 referenceFeatureSpacePoint = make_float2( u4LeftSource, u4RightSource );

        float minFeatureSpaceDistanceToForeground = 999999.0f;
        float minFeatureSpaceDistanceToBackground = 999999.0f;
        
        //
        // foreground
        //
        for ( int i = 0; i < constantMemoryCudaConstraintValues.numForegroundValues; i++ )
        {
            float2 currentFeatureSpaceForegroundPoint      = make_float2( constantMemoryCudaConstraintValues.leftForegroundValues[ i ], constantMemoryCudaConstraintValues.rightForegroundValues[ i ] );
            float  currentFeatureSpaceDistanceToForeground = length( referenceFeatureSpacePoint - currentFeatureSpaceForegroundPoint );
            
            if ( currentFeatureSpaceDistanceToForeground < minFeatureSpaceDistanceToForeground )
            {
                minFeatureSpaceDistanceToForeground = currentFeatureSpaceDistanceToForeground;
            }
        }
        
        //
        // background
        //
       for ( int i = 0; i < constantMemoryCudaConstraintValues.numBackgroundValues; i++ )
        {
            float2 currentFeatureSpaceBackgroundPoint      = make_float2( constantMemoryCudaConstraintValues.leftBackgroundValues[ i ], constantMemoryCudaConstraintValues.rightBackgroundValues[ i ] );
            float  currentFeatureSpaceDistanceToBackground = length( referenceFeatureSpacePoint - currentFeatureSpaceBackgroundPoint );
            
            if ( currentFeatureSpaceDistanceToBackground < minFeatureSpaceDistanceToBackground )
            {
                minFeatureSpaceDistanceToBackground = currentFeatureSpaceDistanceToBackground;
            }
        }
                
        thrust::get< FEATURE_SPACE_DISTANCE_TO_FOREGROUND >( tuple ) = minFeatureSpaceDistanceToForeground;
        thrust::get< FEATURE_SPACE_DISTANCE_TO_BACKGROUND >( tuple ) = minFeatureSpaceDistanceToBackground;
    }
};

extern "C" void CudaInitializeFeatureSpaceDistanceMaps( CudaConstraintValues*           cudaConstraintValues,
                                                        thrust::device_vector< float >& featureSpaceDistanceToForegroundVolumeDevice,
                                                        thrust::device_vector< float >& featureSpaceDistanceToBackgroundVolumeDevice, 
                                                        dim3                            volumeDimensions )
{
    int  numElements          = volumeDimensions.x * volumeDimensions.y * volumeDimensions.z;
    int3 volumeDimensionsInt3 = make_int3( volumeDimensions.x, volumeDimensions.y, volumeDimensions.z );
    
    MOJO_CUDA_SAFE( cudaMemcpyToSymbol( constantMemoryCudaConstraintValues, cudaConstraintValues, sizeof( CudaConstraintValues ) ) );
    
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( 0 ),
                        Index1DToIndex3DFunction( volumeDimensionsInt3 ) ),
                    featureSpaceDistanceToForegroundVolumeDevice.begin(),
                    featureSpaceDistanceToBackgroundVolumeDevice.begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator( 
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( volumeDimensionsInt3 ) ),
                    featureSpaceDistanceToForegroundVolumeDevice.end(),
                    featureSpaceDistanceToBackgroundVolumeDevice.end() ) ),
              
            InitializeFeatureSpaceDistanceMapsFunction() ) );
}
                                                        
#endif
 