#pragma once

#include <string>

namespace Mojo
{
namespace Core
{

template < typename T00 >
std::string ToString( T00 t00 );

template < typename T00, typename T01 >
std::string ToString( T00 t00, T01 t01 );

template < typename T00, typename T01, typename T02 >
std::string ToString( T00 t00, T01 t01, T02 t02 );

template < typename T00, typename T01, typename T02, typename T03 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03 );

template < typename T00, typename T01, typename T02, typename T03, typename T04 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10 );

std::string ToStringHelper( std::string  x );
std::string ToStringHelper( const char*  x );
std::string ToStringHelper( float        x );
std::string ToStringHelper( unsigned int x );
std::string ToStringHelper( int          x );

template < typename T00 >
std::string ToString( T00 t00 )
{
    return std::string(
        ToStringHelper( t00 ) );
}

template < typename T00, typename T01 >
std::string ToString( T00 t00, T01 t01 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) );
}

template < typename T00, typename T01, typename T02 >
std::string ToString( T00 t00, T01 t01, T02 t02 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) );
}

template < typename T00, typename T01, typename T02, typename T03 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) +
        ToStringHelper( t04 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) +
        ToStringHelper( t04 ) +
        ToStringHelper( t05 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) +
        ToStringHelper( t04 ) +
        ToStringHelper( t05 ) +
        ToStringHelper( t06 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) +
        ToStringHelper( t04 ) +
        ToStringHelper( t05 ) +
        ToStringHelper( t06 ) +
        ToStringHelper( t07 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) +
        ToStringHelper( t04 ) +
        ToStringHelper( t05 ) +
        ToStringHelper( t06 ) +
        ToStringHelper( t07 ) +
        ToStringHelper( t08 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) +
        ToStringHelper( t04 ) +
        ToStringHelper( t05 ) +
        ToStringHelper( t06 ) +
        ToStringHelper( t07 ) +
        ToStringHelper( t08 ) +
        ToStringHelper( t09 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10 >
std::string ToString( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10 )
{
    return std::string(
        ToStringHelper( t00 ) +
        ToStringHelper( t01 ) +
        ToStringHelper( t02 ) +
        ToStringHelper( t03 ) +
        ToStringHelper( t04 ) +
        ToStringHelper( t05 ) +
        ToStringHelper( t06 ) +
        ToStringHelper( t07 ) +
        ToStringHelper( t08 ) +
        ToStringHelper( t09 ) +
        ToStringHelper( t10 ) );
}

}

}
