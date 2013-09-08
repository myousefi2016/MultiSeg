#include "userinputhandlers/SketchUserInputHandler.hpp"

#include <QtGui/QMouseEvent>

#include "Engine.hpp"

#include "widgets/RenderingWindow.hpp"

SketchUserInputHandler::SketchUserInputHandler( RenderingWindow* renderingWindow, Engine* engine ) : UserInputHandler( renderingWindow, engine )
{
}

static Qt::MouseButton sCurrentMouseButton = Qt::NoButton;

void SketchUserInputHandler::renderingWindowMousePressEvent( QMouseEvent* mouseEvent )
{
    if ( sCurrentMouseButton == Qt::NoButton )
    {
        sCurrentMouseButton = mouseEvent->button();

        if ( sCurrentMouseButton == Qt::LeftButton )
        {
            GetEngine()->BeginPlaceSeed( SeedType_Foreground );
            GetEngine()->PlaceSeed( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y(), SeedType_Foreground );
        }
        else
        {
            GetEngine()->BeginPlaceSeed( SeedType_Background );
            GetEngine()->PlaceSeed( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y(), SeedType_Background );
        }
    }
}

void SketchUserInputHandler::renderingWindowMouseReleaseEvent( QMouseEvent* mouseEvent )
{
    if ( sCurrentMouseButton == Qt::LeftButton )
    {
        GetEngine()->PlaceSeed( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y(), SeedType_Foreground );
        GetEngine()->EndPlaceSeed( SeedType_Foreground );
    }
    else
    {
        GetEngine()->PlaceSeed( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y(), SeedType_Background );
        GetEngine()->EndPlaceSeed( SeedType_Background );
    }

    sCurrentMouseButton = Qt::NoButton;
}

void SketchUserInputHandler::renderingWindowMouseMoveEvent( QMouseEvent* mouseEvent )
{
    if ( sCurrentMouseButton == Qt::LeftButton )
    {
        GetEngine()->PlaceSeed( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y(), SeedType_Foreground );
    }
    else
    {
        GetEngine()->PlaceSeed( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y(), SeedType_Background );
    }
}

void SketchUserInputHandler::renderingWindowMouseWheelEvent( QWheelEvent* wheelEvent )
{
}

void SketchUserInputHandler::renderingWindowKeyPressEvent( QKeyEvent* keyEvent )
{
}

void SketchUserInputHandler::renderingWindowKeyReleaseEvent( QKeyEvent* keyEvent )
{
}
