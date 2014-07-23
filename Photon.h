#include "R3_vector.h"

#ifndef PHOTON_H
#define PHOTON_H

class Photon
{
public:
	Photon( ):position( 0, 0, 0 ), direction( 0, 0, 0 ) { };
	Photon( R3_vector p, float r_pwr, float g_pwr, float b_pwr, R3_vector d ):position( p ), 
			r_power(r_pwr), g_power(g_pwr), b_power(b_pwr), direction( d ) { };

	R3_vector get_position( ) { return position; };
	float get_r_power( ) { return r_power; };
	float get_g_power( ) { return g_power; };
	float get_b_power( ) { return b_power; };
	void scale_r_power( float scale ) { r_power*=scale; };
	void scale_g_power( float scale ) { g_power*=scale; };
	void scale_b_power( float scale ) { b_power*=scale; };
	R3_vector get_direction( ) { return direction; };
	short get_plane( ) { return plane; };

	void set_plane( short p ) { plane=p;};

private:
	R3_vector position;
	float r_power, g_power, b_power;
	R3_vector direction;
	short plane;
};
#endif