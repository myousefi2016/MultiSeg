#ifndef GPUSEG_RENDERSTRATEGY_HPP
#define GPUSEG_RENDERSTRATEGY_HPP

#include "content/Ref.hpp"

#include "rendering/renderstrategies/RenderStrategy.hpp"

#include "VolumeDesc.hpp"

namespace content
{
class Inventory;
}

namespace rendering
{

class Scene;
class Camera;
class Material;

}

class Engine;

enum SeedType
{
    SeedType_Background,
    SeedType_Foreground
};

class GPUSegRenderStrategy : public rendering::RenderStrategy
{

public:
    GPUSegRenderStrategy();
    virtual ~GPUSegRenderStrategy();

    virtual void SetEngine                        ( Engine* engine );
    virtual void SetLeftSourceVolumeTexture       ( rendering::rtgi::Texture* texture );
    virtual void SetRightSourceVolumeTexture      ( rendering::rtgi::Texture* texture );
    virtual void SetCurrentLevelSetVolumeTexture  ( rendering::rtgi::Texture* texture );
    virtual void SetFrozenLevelSetVolumeTexture   ( rendering::rtgi::Texture* texture );
    virtual void SetActiveElementsVolumeTexture   ( rendering::rtgi::Texture* texture );
    virtual void SetConstraintVolumeTexture       ( rendering::rtgi::Texture* texture );

    virtual void LoadVolume( const VolumeDesc& volumeDesc );
    virtual void UnloadVolume();

    virtual void Update( content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera, double timeDeltaSeconds );
    virtual void Render( content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera );

    virtual void BeginPlaceSeed( SeedType seedType );
    virtual void PlaceSeed( int screenX, int screenY, SeedType seedType );
    virtual void EndPlaceSeed( SeedType seedType );

private:
    math::Vector3 ComputeIntersectionWithCuttingPlane( math::Vector3 virtualScreenCoordinates );
};

#endif