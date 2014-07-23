#include "R3_vector.h"
#include <iostream>
#include <cmath>

std::ostream& operator<<( std::ostream& output, const R3_vector& vec )
{
	output << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
	return output;
}

R3_vector operator +( const R3_vector& vec1, const R3_vector& vec2 )
{
	R3_vector vec3( vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z );
	return vec3;
}

R3_vector operator -( const R3_vector& vec1, const R3_vector& vec2 )
{
	R3_vector vec3( vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z );
	return vec3;
}

R3_vector operator *( const float& scale, const R3_vector& vec )
{
	R3_vector svec( scale*vec.x, scale*vec.y, scale*vec.z );
    return svec;
}

float R3_vector::dot( const R3_vector& vec )
{
	float dotp=x*vec.x+y*vec.y+z*vec.z;
    return dotp;
}

R3_vector R3_vector::cross( const R3_vector& vec )
{
	R3_vector cross_p( y*vec.z-z*vec.y, z*vec.x-x*vec.z, x*vec.y-y*vec.x );
    return cross_p;
}

float R3_vector::norm(  )
{
	float length=sqrt( x*x+y*y+z*z );
	return length;
}

R3_vector R3_vector::normalized( )
{
     R3_vector unit( x/(this->norm( )), y/(this->norm( )), z/(this->norm( )) );
     return unit;
}

void R3_vector::set( float xc, float yc, float zc )
{
    x = xc;
    y = yc;
    z = zc;
}

void R3_vector::shift( char axis, float amount )
{
    if( axis == 'x' || axis == 'X' )
    {
        x = x + amount;
    }
    if( axis == 'y' || axis == 'Y' )
    {
        y = y + amount;
    }
    if( axis == 'z' || axis == 'Z' )
    {
        z = z + amount;
    }
}

void R3_vector::rotate( char axis, float angle )
{
    if( axis == 'x' || axis == 'X' )
    {
        float y2 = y*cos(angle) - z*sin(angle);
        z = y*sin(angle) + z*cos(angle);
        y = y2;
    }
    if( axis == 'y' || axis == 'Y' )
    {
        float x2 = x*cos(angle) + z*sin(angle);
        z = -x*sin(angle) + z*cos(angle);
        x = x2;
    }
    if( axis == 'z' || axis == 'Z' )
    {
        float x2 = x*cos(angle) - y*sin(angle);
        y = x*sin(angle) + y*cos(angle);
        x = x2;
    }
}

float* R3_vector::get_coords( ) const
{
    float* array = new float[ 3 ];
    array[ 0 ] = x;
    array[ 1 ] = y;
    array[ 2 ] = z;
    return array;
}

