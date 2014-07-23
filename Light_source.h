#include "R3_vector.h"
#include <cmath>
#include <math.h>
#include <cstdlib>

#ifndef LIGHTSOURCE_H
#define LIGHTSOURCE_H

class Light_source
{
public:
	Light_source( R3_vector c, R3_vector h, R3_vector v, float i, float r_pwr, float g_pwr, float b_pwr, char t ):corner( c ), 
				  horiz( h ), vert( v ), intensity( i ), r_power(r_pwr), g_power(g_pwr), b_power(b_pwr) , type( t )
	{ area=(get_ll_corner( )-get_ul_corner( )).norm( )*(get_ll_corner( )-get_lr_corner( )).norm( ); };

	float get_intensity( ) { return intensity; };
	char get_type( ) { return type; };
	float get_r_power( ) { return area*r_power; };
	float get_g_power( ) { return area*g_power; };
	float get_b_power( ) { return area*b_power; };
	float get_total_power( ) { return (get_r_power( )+get_g_power( )+get_b_power( )); };
	R3_vector get_location( ) 
	{
		float param1=1.0*rand( )/RAND_MAX;
		float param2=1.0*rand( )/RAND_MAX;

		return corner+param1*horiz+param2*vert;
	}
	R3_vector get_center( )
	{
		return corner+0.5*horiz+0.5*vert;
	}
	R3_vector get_ll_corner( ) { return corner; };
	R3_vector get_ul_corner( ) { return corner+vert; };
	R3_vector get_lr_corner( ) { return corner+horiz; };
	R3_vector get_ur_corner( ) { return corner+horiz+vert; };
	float get_area( ) { return area; };

private:
	R3_vector corner;
	R3_vector horiz;
	R3_vector vert;
	float intensity;
	float area;
	float r_power, g_power, b_power;
	char type;
};
#endif