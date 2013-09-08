#include "ToString.hpp"

#include <boost/lexical_cast.hpp>

namespace Mojo
{
namespace Core
{

std::string ToStringHelper( std::string x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( const char* x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( float x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( unsigned int x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( int x )
{
    return boost::lexical_cast< std::string >( x );
}

}

}
