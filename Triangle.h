#include <iostream>
#include "R3_vector.h"
#include "Primitive.h"

#ifndef TRIANGLE_H
#define TRIANGLE_H

class Triangle:public Primitive
{
public:
        Triangle( ):vertex1( 0, 0, 0 ), vertex2( 1, 0, 0 ), vertex3( 0, 1, 0 ) { };
        Triangle( R3_vector v1, R3_vector v2, R3_vector v3 ):vertex1( v1 ), vertex2( v2 ), vertex3( v3 ) { };

        friend std::ostream& operator<<( std::ostream&, const Triangle& );

	R3_vector get_vertex1( ) { return vertex1; };
	R3_vector get_vertex2( ) { return vertex2; };
	R3_vector get_vertex3( ) { return vertex3; };

private:
        R3_vector vertex1, vertex2, vertex3;
};
#endif
