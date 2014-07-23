#include <iostream>

#ifndef R3_VECTOR_H
#define R3_VECTOR_H

class R3_vector
{
public:
	R3_vector( ):x( 0 ), y( 0 ), z( 0 ) { }; 
	R3_vector( float x, float y, float z ):x( x ), y( y ), z( z ) { };

	friend std::ostream& operator<<( std::ostream&, const R3_vector& );
	friend R3_vector operator+( const R3_vector&, const R3_vector& );
	friend R3_vector operator-( const R3_vector&, const R3_vector& );
	friend R3_vector operator*( const float&, const R3_vector& );
	
	float dot( const R3_vector& ); 
	R3_vector cross( const R3_vector& );
	float norm( );
	R3_vector normalized( );

	void set( float x, float y, float z );
	void shift( char axis, float amount );
    void rotate( char axis, float angle );
    float* get_coords( ) const;

	float get_x_comp( ) { return x; };
	float get_y_comp( ) { return y; };
	float get_z_comp( ) { return z; };

private:
    	float x;
    	float y;
    	float z;
};
#endif
