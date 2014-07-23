#include <iostream>
#include "R3_vector.h"
#include "Primitive.h"

#ifndef SPHERE_H
#define SPHERE_H

class Sphere:public Primitive
{
public:
	Sphere( ):center( 0, 0, 0 ), radius( 1 ) { };
	Sphere( R3_vector c, float r ):center( c ), radius( r ) { };
	
	friend std::ostream& operator<<( std::ostream&, const Sphere& );

	R3_vector get_center( ) { return center; };
	float get_radius( ) { return radius; };

private:
	R3_vector center;
	float radius;
};
#endif	
