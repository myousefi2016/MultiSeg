#pragma once

#include <string>

#include "ToString.hpp"

namespace Mojo
{
namespace Core
{
template < typename T00 >
void Printf( T00 t00 );

template < typename T00, typename T01 >
void Printf( T00 t00, T01 t01 );

template < typename T00, typename T01, typename T02 >
void Printf( T00 t00, T01 t01, T02 t02 );

template < typename T00, typename T01, typename T02, typename T03 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03 );

template < typename T00, typename T01, typename T02, typename T03, typename T04 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10 );

void PrintfHelper( std::string string );

template < typename T00 >
void Printf( T00 t00 )
{
    PrintfHelper( ToString( t00 ) );
}

template < typename T00, typename T01 >
void Printf( T00 t00, T01 t01 )
{
    PrintfHelper( ToString( t00, t01 ) );
}

template < typename T00, typename T01, typename T02 >
void Printf( T00 t00, T01 t01, T02 t02 )
{
    PrintfHelper( ToString( t00, t01, t02 ) );
}

template < typename T00, typename T01, typename T02, typename T03 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03 )
{
    PrintfHelper( ToString( t00, t01, t02, t03 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10 ) );
}

}
}
