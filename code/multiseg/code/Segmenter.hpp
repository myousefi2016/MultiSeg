#ifndef SEGMENTER_HPP
#define SEGMENTER_HPP

#include <builtin_types.h>

#include <cudpp.h>

#include "core/RefCounted.hpp"

#include "container/List.hpp"
#include "container/Array.hpp"

#include "rendering/rtgi/Texture.hpp"

#include "Thrust.hpp"
#include "VolumeDesc.hpp"
#include "Engine.hpp"
#include "cuda/CudaTypes.hpp"

namespace math
{
    class Vector3;
}

namespace rendering
{
namespace rtgi
{
    class VertexBuffer;
    class PixelBuffer;
    class FrameBufferObject;
    class Effect;
}
}

class Engine;

class Segmenter : public core::RefCounted
{
public:
    Segmenter();
    ~Segmenter();

    void                                SetEngine( Engine* engine );
    void                                SetAutomaticParameterAdjustEnabled( bool );

    VolumeDesc                          LoadVolume( const VolumeDesc& volumeDesc );
    void                                UnloadVolume();

    VolumeDesc                          GetSaveableSegmentation();
                                  
    void                                BeginInitializeSeed();
    void                                InitializeSeed( const math::Vector3& seedCoordinates, unsigned int sphereSize );
    void                                EndInitializeSeed();

    void                                BeginInitializeBackgroundConstraint();
    void                                InitializeBackgroundConstraint( const math::Vector3& seedCoordinates, unsigned int sphereSize );
    void                                EndInitializeBackgroundConstraint();

    void                                ClearCurrentSegmentation();
    void                                ClearAllSegmentations();
    void                                FreezeCurrentSegmentation();

    void                                FinishedSegmentationSession();
                                  
    void                                RequestUpdateSegmentation( double timeDeltaSeconds );
    void                                RequestPlaySegmentation();
    void                                RequestPauseSegmentation();
                                  
    bool                                IsSegmentationInitialized();
    bool                                IsSegmentationFinished();
    bool                                IsSegmentationInProgress();
    bool                                IsAutomaticParameterAdjustEnabled();
                                  
    rendering::rtgi::Texture*           GetLeftSourceVolumeTexture();
    rendering::rtgi::Texture*           GetRightSourceVolumeTexture();
    rendering::rtgi::Texture*           GetCurrentLevelSetVolumeTexture();
    rendering::rtgi::Texture*           GetFrozenLevelSetVolumeTexture();
    rendering::rtgi::Texture*           GetActiveElementsVolumeTexture();
    rendering::rtgi::Texture*           GetConstraintVolumeTexture();

    size_t                              GetActiveElementCount();
    VolumeDesc                          GetVolumeDesc();
                                  
private:
    VolumeDesc                          LoadVolumeHostPadded( const VolumeDesc& volumeDesc );
    VolumeDesc                          UnloadVolumeHostPadded( const VolumeDesc& volumeDesc );

    // Unpads volumeDesc according to its difference in size (padding) from mVolumeBeforePaddingDesc
    VolumeDesc                            UnpadVolume( const VolumeDesc& volumeDesc );

    void                                InitializeFeatureSpaceDistanceMaps();

    void                                UpdateRenderingTextures();
    void                                ClearTexture( rendering::rtgi::Texture* texture );
    void                                TagTextureSparse( rendering::rtgi::Texture* texture, float tagValue );

    void                                SwapLevelSetBuffers();

    bool                                ShouldRestartSegmentation();
    bool                                ShouldUpdateSegmentation();
    bool                                ShouldOptimizeForFewActiveVoxels();
    bool                                ShouldInitializeActiveElements();
    bool                                ShouldInitializeCoordinates();
    bool                                ShouldUpdateRenderingTextures();

    void                                UpdateHostStateBeforeRequestUpdate();
    void                                UpdateHostStateAfterRequestUpdate( double timeDeltaSeconds );

    void                                UpdateHostStateRestartSegmentation();

    void                                UpdateHostStateBeforeUpdateSegmentation();
    void                                UpdateHostStateAfterUpdateSegmentation();
    void                                UpdateHostStateDoNotUpdateSegmentation();

    void                                UpdateHostStateBeforeUpdateSegmentationIteration();
    void                                UpdateHostStateAfterUpdateSegmentationIteration( const CudaTagElement* levelSetExportBuffer );

    void                                UpdateHostStateOptimizeForFewActiveVoxels();
    void                                UpdateHostStateOptimizeForManyActiveVoxels();

    void                                UpdateHostStateInitializeActiveElements();
    void                                UpdateHostStateInitializeCoordinates();

    rendering::rtgi::TexturePixelFormat GetTexturePixelFormat( const VolumeDesc& volumeDesc );
    
    const char*                         GetCudaLeftSourceTexture  ( const VolumeDesc& volumeDesc );
    const char*                         GetCudaRightSourceTexture ( const VolumeDesc& volumeDesc );

    unsigned int                        GetNumActiveElementsAligned();

    void                                BeforeLoadVolumeDebug( const VolumeDesc& volumeDesc );
    void                                AfterLoadVolumeDebug();

    void                                WriteSegmentationToFile();
    void                                ComputeSegmentationAccuracy( const CudaTagElement* levelSetExportBuffer );
    void                                PrintSegmentationDetails();

    void                                CalculateSegmentationParameters();

    void                                CollectCurrentLevelSetVoxels( container::Array< float >& leftOutCollectedValues,
                                                                      container::Array< float > &rightOutCollectedValues );
    void                                CollectCurrentConstraintVoxels( container::Array< float >& leftOutForegroundValues,
                                                                        container::Array< float >& rightOutForegroundValues,
                                                                        container::Array< float >& leftOutBackgroundValues,
                                                                        container::Array< float >& rightOutBackgroundValues );

    // device data global memory
    CudaLevelSetElement*             mLevelSetVolumeDeviceX;
    CudaLevelSetElement*             mLevelSetVolumeDeviceY;
    CudaLevelSetElement*             mLevelSetVolumeDeviceRead;
    CudaLevelSetElement*             mLevelSetVolumeDeviceWrite;

    CudaCompactElement*              mCoordinatesVolumeDevice;
    CudaCompactElement*              mValidElementsVolumeDevice;

    CudaTagElement*                  mTagVolumeDevice;

    thrust::device_vector< float >   mFeatureSpaceDistanceToBackgroundVolumeDevice;
    thrust::device_vector< float >   mFeatureSpaceDistanceToForegroundVolumeDevice;

    cudaArray*                       mLeftSourceVolumeArray3DDevice;
    cudaArray*                       mRightSourceVolumeArray3DDevice;

    size_t*                          mNumValidActiveElementsDevice;

    // compact
    CUDPPHandle                      mCompactPlanHandle;

    // rtgi objects
    rendering::rtgi::Texture*           mLeftSourceVolumeTexture;
    rendering::rtgi::Texture*           mRightSourceVolumeTexture;

    rendering::rtgi::Texture*           mActiveElementsTexture;
    rendering::rtgi::Texture*           mCurrentLevelSetVolumeTexture;
    rendering::rtgi::Texture*           mFrozenLevelSetVolumeTexture;

    rendering::rtgi::Texture*           mConstraintVolumeTexture;

    rendering::rtgi::PixelBuffer*       mLevelSetExportPixelBuffer;
    rendering::rtgi::PixelBuffer*       mFrozenLevelSetExportPixelBuffer;
    rendering::rtgi::PixelBuffer*       mConstraintExportPixelBuffer;

    rendering::rtgi::VertexBuffer*      mActiveElementsCompactedVertexBuffer;

    rendering::rtgi::FrameBufferObject* mFrameBufferObject;

    rendering::rtgi::Effect*            mTagVolumeEffect;

    // sizes of volumes
    dim3                             mVolumeDimensions;
                               
    int                              mSourceVolumeNumBytes;
    int                              mTagVolumeNumBytes;
    int                              mLevelSetVolumeNumBytes;
    int                              mCompactVolumeNumBytes;

    // simulation state
    size_t                           mNumValidActiveElementsHost;
    int                              mVolumeNumElements;
    int                              mNumIterations;
    float                            mTargetPrevious;
    float                            mMaxDistanceBeforeShrinkPrevious;
    float                            mCurvatureInfluencePrevious;
    float                            mTimeStepPrevious;

    // volume desc
    VolumeDesc                       mVolumeDesc;
    VolumeDesc                         mVolumeBeforePaddingDesc;

    // engine
    Engine*                          mEngine;

    // playback state
    bool                             mLevelSetInitialized;
    bool                             mCoordinatesVolumeInitialized;
    bool                             mComputedSegmentationDetailsOnce;
    bool                             mPaused;
    bool                             mCallEngineOnResumeSegmentation;
    bool                             mCallEngineOnStopSegmentation;
    bool                             mAutomaticParameterAdjustEnabled;
    bool                             mCurrentlySketching;

#ifdef COMPUTE_PERFORMANCE_METRICS

    // counters and buffers
    CudaLevelSetElement*       mLevelSetVolumeDummy1;
    CudaLevelSetElement*       mLevelSetVolumeDummy2;
    CudaTagElement*            mTagVolumeDummy;
    CudaCompactElement*        mCompactVolumeDummy;
    container::Array< int >    mActiveTileCounter;
    container::Array< int >    mActiveVoxelCounter;
    container::Array< double > mLevelSetUpdateTimer;
    container::Array< double > mOutputNewActiveVoxelsTimer;
    container::Array< double > mInitializeActiveVoxelsConditionalMemoryWriteTimer;
    container::Array< double > mInitializeActiveVoxelsUnconditionalMemoryWriteTimer;
    container::Array< double > mWriteAndCompactEntireVolumeTimer;
    container::Array< double > mFilterDuplicatesTimer;
    container::Array< double > mCompactTimer;
    container::Array< double > mClearTagVolumeTimer;
    container::Array< double > mClearValidVolumeTimer;

#endif

};

#endif
