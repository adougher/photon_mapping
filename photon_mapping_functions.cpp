#include "photon_mapping_functions.h"
#include "Image.h"
#include "Sphere.h"
#include "Triangle.h"
#include "helper_structs.h"
#include "Photon.h"
#include <vector>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <cmath>
#include <math.h>
#include <queue>
#include <cstdlib>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <ANN.h>

#include "mesh.h"
#include "trackballhandler.h"
#include "program.h"
#include "buffer.h"
#include "glutwrapper.h"
#include "menucreator.h"

using namespace std;
using namespace glm;
using namespace EZGraphics;

float s=.25;

vec3 mod289(vec3 x) {
  return x - vec3(floor(x[0] * (1.0 / 289.0)) * 289.0, floor(x[1] * (1.0 / 289.0)) * 289.0,
				  floor(x[2] * (1.0 / 289.0)) * 289.0);
}

vec4 mod289(vec4 x) {
  return x - vec4(floor(x[0] * (1.0 / 289.0)) * 289.0, floor(x[1] * (1.0 / 289.0)) * 289.0,
				  floor(x[2] * (1.0 / 289.0)) * 289.0, floor(x[3] * (1.0 / 289.0)) * 289.0);
}

vec4 permute(vec4 x) {
	vec4 y=vec4( ((x[0]*34.0)+1.0)*x[0], ((x[1]*34.0)+1.0)*x[1], ((x[2]*34.0)+1.0)*x[2],
				 ((x[3]*34.0)+1.0)*x[3] );
    return mod289(y);
}

vec4 taylorInvSqrt(vec4 r)
{
  return vec4(1.79284291400159, 1.79284291400159, 1.79284291400159, 1.79284291400159) - 
	     vec4( 0.85373472095314 * r[0], 0.85373472095314 * r[1], 0.85373472095314 * r[2], 0.85373472095314 * r[3] );
}

float snoise(vec3 v)
{ 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = vec3(floor(v[0] + dot(v, vec3(C[1],C[1],C[1])) ), floor(v[1] + dot(v, vec3(C[1],C[1],C[1])) ),
				 floor(v[2] + dot(v, vec3(C[1],C[1],C[1])) ) );
  vec3 x0 =   v - i + vec3(dot(i, vec3(C[0],C[0],C[0])), dot(i, vec3(C[0],C[0],C[0])), dot(i, vec3(C[0],C[0],C[0])) );

// Other corners
  vec3 g = step(vec3(x0[1], x0[2], x0[0]), vec3(x0[0], x0[1], x0[2]));
  vec3 l = vec3(1.0,1.0,1.0) - g;
  vec3 i1 = vec3(std::min(g[0],l[2]),std::min(g[1],l[0]),std::min(g[2],l[1]));//min( g.xyz, l.zxy );
  vec3 i2 = vec3(std::max(g[0],l[2]),std::max(g[1],l[0]),std::max(g[2],l[1]));// max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + vec3(C[0],C[0],C[0]);
  vec3 x2 = x0 - i2 + vec3(C[1],C[1],C[1]); // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - vec3(D[1],D[1],D[1]);      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             vec4(i[2],i[2],i[2],i[2]) + vec4(0.0, i1[2], i2[2], 1.0 ))
           + vec4(i[1],i[1],i[1],i[1]) + vec4(0.0, i1[1], i2[1], 1.0 )) 
           + vec4(i[0],i[0],i[0],i[0]) + vec4(0.0, i1[0], i2[0], 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_*vec3(D[3],D[1],D[2])-vec3(D[0],D[2],D[0]);

  vec4 j = p - vec4(49.0 * floor(p[0] * ns[2] * ns[2]), 49.0 * floor(p[1] * ns[2] * ns[2]),
					49.0 * floor(p[2] * ns[2] * ns[2]), 49.0 * floor(p[3] * ns[2] * ns[2]) );  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = vec4(floor(j[0] - 7.0 * x_[0]),floor(j[1] - 7.0 * x_[1]),
				 floor(j[2] - 7.0 * x_[2]),floor(j[3] - 7.0 * x_[3]) );    // mod(j,N)

  vec4 x = x_ *ns.x + vec4(ns[1],ns[1],ns[1],ns[1]);
  vec4 y = y_ *ns.x + vec4(ns[1],ns[1],ns[1],ns[1]);
  vec4 h = vec4(1.0 - abs(x[0]) - abs(y[0]), 1.0 - abs(x[1]) - abs(y[1]),
				1.0 - abs(x[2]) - abs(y[2]), 1.0 - abs(x[3]) - abs(y[3]) );

  vec4 b0 = vec4( x[0], x[1], y[0], y[1] );
  vec4 b1 = vec4( x[2], x[3], y[2], y[3] );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = vec4( floor(b0[0])*2.0 + 1.0, floor(b0[1])*2.0 + 1.0,
				  floor(b0[2])*2.0 + 1.0, floor(b0[3])*2.0 + 1.0 );
  vec4 s1 = vec4( floor(b1[0])*2.0 + 1.0, floor(b1[1])*2.0 + 1.0,
				  floor(b1[2])*2.0 + 1.0, floor(b1[3])*2.0 + 1.0 );
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = vec4(b0[0],b0[2],b0[1],b0[3]) + vec4(s0[0]*sh[0],s0[2]*sh[0],s0[1]*sh[1],s0[3]*sh[1]);
  vec4 a1 = vec4(b1[0],b1[2],b1[1],b1[3]) + vec4(s1[0]*sh[2],s0[2]*sh[2],s0[1]*sh[3],s0[3]*sh[3]);

  vec3 p0 = vec3(a0[0],a0[1],h.x);
  vec3 p1 = vec3(a0[2],a0[3],h.y);
  vec3 p2 = vec3(a1[0],a1[1],h.z);
  vec3 p3 = vec3(a1[2],a1[3],h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = vec4( std::max(0.6 - dot(x0,x0),0.0), std::max(0.6 - dot(x1,x1),0.0), 
				 std::max(0.6 - dot(x2,x2),0.0), std::max(0.6 - dot(x3,x3),0.0) );
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

void pmf::read_input_file( int& resolution_x, int& resolution_y, float* viewpoint, float* screen_lower_left_corner, float* screen_horizontal_vector, 
						   float* screen_vertical_vector, std::vector<Light_source>& lights, float& ambient_light_intensity, 
						   std::vector<Sphere>& spheres, std::vector<Triangle>& triangles )
{
  	std::ifstream ifs("C:/Users/Andrew/Documents/School/Computer_Science/Advanced_Graphics/Final_Project/photon_inputs/test11.txt");
  	assert(ifs);

  	int number_of_primitives;
	int number_of_lights;

  	ifs >> resolution_x >> resolution_y;
  	ifs >> viewpoint[0] >> viewpoint[1] >> viewpoint[2];
  	ifs >> screen_lower_left_corner[0] >> screen_lower_left_corner[1] >> screen_lower_left_corner[2];
  	ifs >> screen_horizontal_vector[0] >> screen_horizontal_vector[1] >> screen_horizontal_vector[2];
  	ifs >> screen_vertical_vector[0] >> screen_vertical_vector[1] >> screen_vertical_vector[2];
	ifs >> number_of_lights;
	float c1, c2, c3, h1, h2, h3, v1, v2, v3, intensity;
	float r_power, g_power, b_power;
	char type;
	for( int i=0; i<number_of_lights; i++ )
	{
		ifs >> c1 >> c2 >> c3 >> h1 >> h2 >> h3 >> v1 >> v2 >> v3 >> intensity >> r_power >> g_power >> b_power >> type;
		lights.push_back( Light_source(R3_vector(c1,c2,c3),R3_vector(h1,h2,h3),R3_vector(v1,v2,v3),intensity, 
						  r_power, g_power, b_power, type ) );
	}

	ifs >> number_of_primitives;

  	// save all this info to your datastructures or global variables here

  	for ( int i=0; i<number_of_primitives; i++ )
    {
     		char primitive_type;
      		ifs >> primitive_type;
      		switch(primitive_type)
        	{
        	case 's':
        	case 'S':
          	{
            	float center[3];
            	float radius;
            	float k_diffuse[3];
            	float k_specular[3];
				float k_transmission[3];
				float ref_index;
            	float n_specular;
				short type;

            	ifs >> center[0] >> center[1] >> center[2];
            	ifs >> radius;
				ifs >> k_diffuse[0] >> k_diffuse[1] >> k_diffuse[2];
            	ifs >> k_specular[0] >> k_specular[1] >> k_specular[2];
				ifs >> k_transmission[0] >> k_transmission[1] >> k_transmission[2];
				ifs >> ref_index;
            	ifs >> n_specular;
				ifs >> type;

            	// add the sphere to your datastructures (primitive list, sphere list or such) here
				hs::material material_data( k_diffuse[0], k_diffuse[1], k_diffuse[2], k_specular[0], k_specular[1], k_specular[2],
											k_transmission[0], k_transmission[1], k_transmission[2], ref_index, n_specular, type );
				R3_vector temp_vector( center[0], center[1], center[2] );
				Sphere new_sphere( temp_vector, radius );
				new_sphere.set_material( material_data );
				spheres.push_back( new_sphere );	
          	}
          	break;
        	case 'T':
        	case 't':
          	{
            	float a1[3];
            	float a2[3];
            	float a3[3];
            	float k_diffuse[3];
            	float k_specular[3];
				float k_transmission[3];
				float ref_index;
            	float n_specular;
				short type;

				ifs >> a1[0] >> a1[1] >> a1[2];
            	ifs >> a2[0] >> a2[1] >> a2[2];
            	ifs >> a3[0] >> a3[1] >> a3[2];
            	ifs >> k_diffuse[0] >> k_diffuse[1] >> k_diffuse[2];
            	ifs >> k_specular[0] >> k_specular[1] >> k_specular[2];
				ifs >> k_transmission[0] >> k_transmission[1] >> k_transmission[2];
				ifs >> ref_index;
            	ifs >> n_specular;
				ifs >> type;

            	// add the triangle to your datastructure (primitive list, sphere list or such) here
				hs::material material_data( k_diffuse[0], k_diffuse[1], k_diffuse[2], k_specular[0], k_specular[1], k_specular[2],
											k_transmission[0], k_transmission[1], k_transmission[2], ref_index, n_specular, type );
                R3_vector temp_vector1( a1[0], a1[1], a1[2] );
				R3_vector temp_vector2( a2[0], a2[1], a2[2] );
				R3_vector temp_vector3( a3[0], a3[1], a3[2] );
                Triangle new_triangle( temp_vector1, temp_vector2,temp_vector3 );
                new_triangle.set_material( material_data );
                triangles.push_back( new_triangle );
          	}
          	break;
        	default:
          		assert(0);
        	}
    	}
}

float pmf::find_intersection_parameter_sphere( R3_vector& origin, R3_vector& direction, Sphere& sphere )
{
	float t=-1;

	float A=direction.dot( direction );
	float B=2.0*((origin-sphere.get_center( )).dot( direction ));
	float C=(origin-sphere.get_center( )).dot(origin-sphere.get_center( ))-(sphere.get_radius( ))*(sphere.get_radius( ));

	float discriminant=B*B-4.0*A*C;

	if( discriminant == 0 )
	{
		float t1=-(B/(2.0*A));
		if( t1 > 0 )
		{
			t=t1;
		}
	}
	if( discriminant > 0 )
    {
		float t1=(-B+sqrt(discriminant))/(2.0*A);
		float t2=(-B-sqrt(discriminant))/(2.0*A);
		if( t1*t2 < 0 )
		{
			if( t2 < t1 )
			{
                t=t1;
			}
			else
			{
				t=t2;
			}
		}
		else
		{
			if( t1 > 0 && t2 > 0 )
			{
				if( t1 < t2 )
				{
					t=t1;
				}
				else
				{
					t=t2;
				}
			}
		}
	}
	return t;
}

float pmf::find_intersection_parameter_triangle( R3_vector& origin, R3_vector& direction, Triangle& triangle )
{
	float t=-1;
	R3_vector intersection;
	R3_vector intersection_temp;

	R3_vector normal=(triangle.get_vertex2( )-triangle.get_vertex1( )).cross(triangle.get_vertex3( )-triangle.get_vertex1( ));
	float t1=((triangle.get_vertex1( )-origin).dot( normal ))/(direction.dot( normal ));
	
	if( t1 >= 0 )
	{
		intersection_temp=origin+t1*direction;
		R3_vector test_vec1=(triangle.get_vertex1( )-intersection_temp).cross(triangle.get_vertex2( )-intersection_temp);
		R3_vector test_vec2=(triangle.get_vertex2( )-intersection_temp).cross(triangle.get_vertex3( )-intersection_temp);
		R3_vector test_vec3=(triangle.get_vertex3( )-intersection_temp).cross(triangle.get_vertex1( )-intersection_temp);
		if( test_vec1.dot(test_vec2) >= 0 && test_vec2.dot(test_vec3) >= 0 && test_vec3.dot(test_vec1) >= 0 )
		{
			t=t1;
		}
	}
	return t;
}

char pmf::get_power_expon( float num )
{
	char count=-1;
	while( num < 256 )
	{
		count++;
		num*=2;
	}
	return count;
}

void pmf::emit_photons( std::vector<Light_source>& lights, unsigned long int total_photons, bool do_direct, std::vector<Photon*>& direct_photons, bool do_indirect, 
						std::vector<Photon*>& indirect_photons, bool do_caust, std::vector<Photon*>& caust_photons, std::vector<Sphere>& spheres, 
						std::vector<Triangle>& triangles, bool do_ray_trace, bool one_map, bool do_color_bleeding )
{
	int light_index;
	float x=1.0;
	float y=1.0;
	float z=1.0;

	int s_size=spheres.size( );
	int t_size=triangles.size( );
	int index=0;
	char type='a';
	int final_index=0;

	float t=-1;
	float t1;
	R3_vector origin;
	R3_vector direction;
	R3_vector hemisphere_direction;
	R3_vector intersection;

	bool hits;
	bool is_direct, is_indirect, is_caust;
	unsigned int emissive_photons=0;

	while( emissive_photons < total_photons )
	{
		hits=0;
		x=1.0;
		y=1.0;
		z=1.0;
		t=-1;
		type='a';
		final_index=0;
		index=0;
		is_direct=0;
		is_indirect=0;
		is_caust=0;

		float total_power=0;
		for( unsigned int i=0; i < lights.size( ); i++ )
		{
			total_power+=lights[i].get_total_power( );
		}
		float prob=1.0*rand( )/RAND_MAX;
		float which_light=0;
		for( unsigned int i=0; i < lights.size( ); i++ )
		{
			which_light+=lights[i].get_total_power( );
			if( prob < which_light )
			{
				light_index=i;
				break;
			}
		}

		hemisphere_direction=(lights[light_index].get_ll_corner( )-lights[light_index].get_lr_corner( )).
								cross(lights[light_index].get_ll_corner( )-lights[light_index].get_ul_corner( )).normalized( );
		if( (triangles[0].get_vertex1( )-lights[light_index].get_center( )).dot( hemisphere_direction ) < 0 )
		{
			hemisphere_direction=-1.0*hemisphere_direction;
		}
		origin=lights[light_index].get_location( );
		if( lights[light_index].get_type( )=='d' )
		{
			direction=hemisphere_direction;
			while( index < s_size )
			{
				t1=find_intersection_parameter_sphere( origin, direction, spheres[index] );
				if( t1 > 0 )
				{
					if( t < 0 )
					{
						t=t1;
						final_index=index;
						type='s';
					}
					else
					{
						if( t1 < t )
						{	
							t=t1;	
							final_index=index;
							type='s';
						}
					}
				}
				index++;
			}
			index=0;
			while( index < t_size )
			{
				t1=find_intersection_parameter_triangle( origin, direction, triangles[index] );
				if( t1 > 0 )
				{
					if( t < 0 )
					{
						t=t1;
						final_index=index;
						type='t';
					}
					else
					{
						if( t1 < t )
						{
							t=t1;
							final_index=index;
							type='t';
						}
					}
				}
				index++;
			}
		}
		if( lights[light_index].get_type( )=='a' )
		{
			while( x*x+y*y+z*z > 1 || t==-1 )
			{
				x=2.0*rand( )/RAND_MAX-1.0;
				y=2.0*rand( )/RAND_MAX-1.0;
				z=2.0*rand( )/RAND_MAX-1.0;

				if( x*x+y*y+z*z <= 1 )
				{
					direction.set(x,y,z);
					direction=direction.normalized( );
					direction=hemisphere_direction+direction;
					direction=direction.normalized( );
		
					while( index < s_size )
					{
						t1=find_intersection_parameter_sphere( origin, direction, spheres[index] );
						if( t1 > 0 )
						{
							if( t < 0 )
							{
								t=t1;
								final_index=index;
								type='s';
							}
							else
							{
								if( t1 < t )
								{	
									t=t1;	
									final_index=index;
									type='s';
								}
							}
						}
						index++;
					}
					index=0;
					while( index < t_size )
					{
						t1=find_intersection_parameter_triangle( origin, direction, triangles[index] );
						if( t1 > 0 )
						{
							if( t < 0 )
							{
								t=t1;
								final_index=index;
								type='t';
							}
							else
							{
								if( t1 < t )
								{
									t=t1;
									final_index=index;
									type='t';
								}
							}
						}
						index++;
					}
				}
			}
		}
		intersection=origin+t*direction;

		emissive_photons++;
		/* float power=1.0/number_of_photons*lights[light_index].get_total_power( )/3.0;
		unsigned char expon=get_power_expon( power );
		unsigned char r_power=int(power*pow( 2.0, expon ));
		unsigned char g_power=int(power*pow( 2.0, expon ));
		unsigned char b_power=int(power*pow( 2.0, expon ));
			
		photons[count]=new Photon( intersection, r_power, g_power, b_power, expon, direction );
		count++; */

		float kd_avg, ks_avg, kt_avg;
		float r_power=1.0*lights[light_index].get_r_power( );
		float g_power=1.0*lights[light_index].get_g_power( );
		float b_power=1.0*lights[light_index].get_b_power( );

		if( type=='t' )
		{
			hemisphere_direction=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
											cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );                

			hemisphere_direction=hemisphere_direction.normalized( );
			if( hemisphere_direction.dot( direction ) > 0 )
			{
				hemisphere_direction=(-1.0)*hemisphere_direction;
			}
			kd_avg=(triangles[final_index].get_material( ).k_dr+triangles[final_index].get_material( ).k_dg+
					triangles[final_index].get_material( ).k_db)/3.0;
			ks_avg=(triangles[final_index].get_material( ).k_sr+triangles[final_index].get_material( ).k_sg+
					triangles[final_index].get_material( ).k_sb)/3.0;
			kt_avg=(triangles[final_index].get_material( ).k_tr+triangles[final_index].get_material( ).k_tg+
					triangles[final_index].get_material( ).k_tb)/3.0;
		}
		if( type=='s' )
		{
			hemisphere_direction=intersection-spheres[final_index].get_center( );                

			hemisphere_direction=hemisphere_direction.normalized( );
			kd_avg=(spheres[final_index].get_material( ).k_dr+spheres[final_index].get_material( ).k_dg+
					spheres[final_index].get_material( ).k_db)/3.0;
			ks_avg=(spheres[final_index].get_material( ).k_sr+spheres[final_index].get_material( ).k_sg+
					spheres[final_index].get_material( ).k_sb)/3.0;
			kt_avg=(spheres[final_index].get_material( ).k_tr+spheres[final_index].get_material( ).k_tg+
					spheres[final_index].get_material( ).k_tb)/3.0;
		}	
		is_direct=1;

		float what=1.0*rand( )/RAND_MAX;
		while( t!=-1 && emissive_photons < total_photons && what < kd_avg+ks_avg+kt_avg )
		{
			if( what < kd_avg )
			{
				if( do_color_bleeding )
				{
					float mean=(r_power+g_power+b_power)/3.0;
					if( type=='s' )
					{
						/* r_power=lights[light_index].get_total_power( )*std::min(spheres[final_index].get_material( ).k_dr,r_power/lights[light_index].get_total_power( ));
						g_power=lights[light_index].get_total_power( )*std::min(spheres[final_index].get_material( ).k_dg,g_power/lights[light_index].get_total_power( ));
						b_power=lights[light_index].get_total_power( )*std::min(spheres[final_index].get_material( ).k_db,b_power/lights[light_index].get_total_power( )); */

						/* r_power*=spheres[final_index].get_material( ).k_dr;
						g_power*=spheres[final_index].get_material( ).k_dg;
						b_power*=spheres[final_index].get_material( ).k_db; */

						r_power=.5*(r_power/mean+spheres[final_index].get_material( ).k_dr);
						g_power=.5*(g_power/mean+spheres[final_index].get_material( ).k_dg);
						b_power=.5*(b_power/mean+spheres[final_index].get_material( ).k_db);

						float baba=total_power/(r_power+g_power+b_power);

						r_power*=baba;
						g_power*=baba;
						b_power*=baba;
						
						/* r_power=(r_power+r_power*spheres[final_index].get_material( ).k_dr)/2;
						g_power=(g_power+g_power*spheres[final_index].get_material( ).k_dg)/2;
						b_power=(b_power+b_power*spheres[final_index].get_material( ).k_db)/2; */
					}
					if( type=='t' )
					{
						/* r_power=lights[light_index].get_total_power( )*std::min(triangles[final_index].get_material( ).k_dr,r_power/lights[light_index].get_total_power( ));
						g_power=lights[light_index].get_total_power( )*std::min(triangles[final_index].get_material( ).k_dg,g_power/lights[light_index].get_total_power( ));
						b_power=lights[light_index].get_total_power( )*std::min(triangles[final_index].get_material( ).k_db,b_power/lights[light_index].get_total_power( )); */

						/* r_power*=triangles[final_index].get_material( ).k_dr;
						g_power*=triangles[final_index].get_material( ).k_dg;
						b_power*=triangles[final_index].get_material( ).k_db; */

						r_power=.5*(r_power/mean+triangles[final_index].get_material( ).k_dr);
						g_power=.5*(g_power/mean+triangles[final_index].get_material( ).k_dg);
						b_power=.5*(b_power/mean+triangles[final_index].get_material( ).k_db);

						float baba=total_power/(r_power+g_power+b_power);

						r_power*=baba;
						g_power*=baba;
						b_power*=baba;

						/* r_power=(r_power+r_power*triangles[final_index].get_material( ).k_dr)/2;
						g_power=(g_power+g_power*triangles[final_index].get_material( ).k_dg)/2;
						b_power=(b_power+b_power*triangles[final_index].get_material( ).k_db)/2; */
					}
				}

				if( !one_map )
				{
					if( is_indirect && t!=-1 && do_indirect )
					{
						if( do_indirect )
						{
							indirect_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
						}
						t=-1;
					}
					if( is_direct && t!=-1 )
					{
						if( do_direct )
						{
							direct_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
						}
						is_direct=0;
						is_indirect=1;
						t=-1;
					}
					if( is_caust && t!=-1 )
					{
						if( do_caust )
						{
							caust_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
						}
						is_caust=0;
						what=5.0;
						t=-1;
					}
				}
				else
				{
					if( emissive_photons < total_photons )
					{
						direct_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
					}
				}

				if( what < 5.0 )
				{
					x=1.0;
					y=1.0;
					z=1.0;
					t=-1;
					index=0;

					type='a';
					final_index=0;		

					while( x*x+y*y+z*z > 1 || t==-1 )
					{
						x=2.0*rand( )/RAND_MAX-1.0;
						y=2.0*rand( )/RAND_MAX-1.0;
						z=2.0*rand( )/RAND_MAX-1.0;

						if( x*x+y*y+z*z <= 1 )
						{
							direction.set(x,y,z);
							direction=direction.normalized( );
							direction=hemisphere_direction+direction;
							direction=direction.normalized( );
							origin=intersection+0.01*direction;
		
							while( index < s_size )
							{
								t1=find_intersection_parameter_sphere( origin, direction, spheres[index] );
								if( t1 > 0 )
								{
									if( t < 0 )
									{
										t=t1;
										final_index=index;
										type='s';
									}
									else
									{
										if( t1 < t )
										{	
											t=t1;	
											final_index=index;
											type='s';
										}
									}
								}
								index++;
							}
							index=0;
							while( index < t_size )
							{
								t1=find_intersection_parameter_triangle( origin, direction, triangles[index] );
								if( t1 > 0 )
								{
									if( t < 0 )
									{
										t=t1;
										final_index=index;
										type='t';
									}
									else
									{
										if( t1 < t )
										{
											t=t1;
											final_index=index;
											type='t';
										}
									}
								}
								index++;
							}
						}
					}
					intersection=origin+t*direction;

					if( type=='t' )
					{
						hemisphere_direction=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
											  cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );                

						hemisphere_direction=hemisphere_direction.normalized( );
						if( hemisphere_direction.dot( direction ) > 0 )
						{
							hemisphere_direction=(-1.0)*hemisphere_direction;
						}
						kd_avg=(triangles[final_index].get_material( ).k_dr+triangles[final_index].get_material( ).k_dg+
								triangles[final_index].get_material( ).k_db)/3.0;
						ks_avg=(triangles[final_index].get_material( ).k_sr+triangles[final_index].get_material( ).k_sg+
								triangles[final_index].get_material( ).k_sb)/3.0;
						kt_avg=(triangles[final_index].get_material( ).k_tr+triangles[final_index].get_material( ).k_tg+
								triangles[final_index].get_material( ).k_tb)/3.0;
					}
					if( type=='s' )
					{
						hemisphere_direction=intersection-spheres[final_index].get_center( );                

						hemisphere_direction=hemisphere_direction.normalized( );
						kd_avg=(spheres[final_index].get_material( ).k_dr+spheres[final_index].get_material( ).k_dg+
								spheres[final_index].get_material( ).k_db)/3.0;
						ks_avg=(spheres[final_index].get_material( ).k_sr+spheres[final_index].get_material( ).k_sg+
								spheres[final_index].get_material( ).k_sb)/3.0;
						kt_avg=(spheres[final_index].get_material( ).k_tr+spheres[final_index].get_material( ).k_tg+
								spheres[final_index].get_material( ).k_tb)/3.0;
					}
					is_indirect=1;
					is_direct=0;
				}
			}
			if( what < kd_avg+ks_avg && what >= kd_avg )
			{
				if( do_color_bleeding )
				{
					float mean=(r_power+g_power+b_power)/3.0;
					if( type=='s' )
					{
						r_power=.5*(r_power/mean+spheres[final_index].get_material( ).k_sr);
						g_power=.5*(g_power/mean+spheres[final_index].get_material( ).k_sg);
						b_power=.5*(b_power/mean+spheres[final_index].get_material( ).k_sb);

						float baba=total_power/(r_power+g_power+b_power);

						r_power*=baba;
						g_power*=baba;
						b_power*=baba;
					}

					if( type=='t' )
					{
						r_power=.5*(r_power/mean+triangles[final_index].get_material( ).k_sr);
						g_power=.5*(g_power/mean+triangles[final_index].get_material( ).k_sg);
						b_power=.5*(b_power/mean+triangles[final_index].get_material( ).k_sb);

						float baba=total_power/(r_power+g_power+b_power);

						r_power*=baba;
						g_power*=baba;
						b_power*=baba;
					}
				}

				t=-1;
				index=0;

				type='a';
				final_index=0;	

				direction=(intersection-origin).normalized( )-(2*((intersection-origin).normalized( )).
						  dot(hemisphere_direction.normalized( )))*(hemisphere_direction).normalized( );
				direction=direction.normalized( );
				origin=intersection+.01*direction;
		
				while( index < s_size )
				{
					t1=find_intersection_parameter_sphere( origin, direction, spheres[index] );
					if( t1 > 0 )
					{
						if( t < 0 )
						{
							t=t1;
							final_index=index;
							type='s';
						}
						else
						{
							if( t1 < t )
							{	
								t=t1;	
								final_index=index;
								type='s';
							}
						}
					}
					index++;
				}
				index=0;
				while( index < t_size )
				{
					t1=find_intersection_parameter_triangle( origin, direction, triangles[index] );
					if( t1 > 0 )
					{
						if( t < 0 )
						{
							t=t1;
							final_index=index;
							type='t';
						}
						else
						{
							if( t1 < t )
							{
								t=t1;
								final_index=index;
								type='t';
							}
						}
					}
					index++;
				}
				intersection=origin+t*direction;

				if( type=='t' )
				{
					hemisphere_direction=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
										  cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );                

					hemisphere_direction=hemisphere_direction.normalized( );
					if( hemisphere_direction.dot( direction ) > 0 )
					{
						hemisphere_direction=(-1.0)*hemisphere_direction;
					}
					kd_avg=(triangles[final_index].get_material( ).k_dr+triangles[final_index].get_material( ).k_dg+
							triangles[final_index].get_material( ).k_db)/3.0;
					ks_avg=(triangles[final_index].get_material( ).k_sr+triangles[final_index].get_material( ).k_sg+
							triangles[final_index].get_material( ).k_sb)/3.0;
					kt_avg=(triangles[final_index].get_material( ).k_tr+triangles[final_index].get_material( ).k_tg+
							triangles[final_index].get_material( ).k_tb)/3.0;
				}
				if( type=='s' )
				{
					hemisphere_direction=intersection-spheres[final_index].get_center( );                

					hemisphere_direction=hemisphere_direction.normalized( );
					kd_avg=(spheres[final_index].get_material( ).k_dr+spheres[final_index].get_material( ).k_dg+
							spheres[final_index].get_material( ).k_db)/3.0;
					ks_avg=(spheres[final_index].get_material( ).k_sr+spheres[final_index].get_material( ).k_sg+
							spheres[final_index].get_material( ).k_sb)/3.0;
					kt_avg=(spheres[final_index].get_material( ).k_tr+spheres[final_index].get_material( ).k_tg+
							spheres[final_index].get_material( ).k_tb)/3.0;
				}
				is_indirect=0;
				is_direct=0;
				is_caust=1;
			}
			if( what < kd_avg+ks_avg+kt_avg && what >= kd_avg+ks_avg )
			{
				if( do_color_bleeding )
				{
					float mean=(r_power+g_power+b_power)/3.0;
					if( type=='s' )
					{
						r_power=.5*(r_power/mean+spheres[final_index].get_material( ).k_tr);
						g_power=.5*(g_power/mean+spheres[final_index].get_material( ).k_tg);
						b_power=.5*(b_power/mean+spheres[final_index].get_material( ).k_tb);

						float baba=total_power/(r_power+g_power+b_power);

						r_power*=baba;
						g_power*=baba;
						b_power*=baba;
					}

					if( type=='t' )
					{
						r_power=.5*(r_power/mean+triangles[final_index].get_material( ).k_tr);
						g_power=.5*(g_power/mean+triangles[final_index].get_material( ).k_tg);
						b_power=.5*(b_power/mean+triangles[final_index].get_material( ).k_tb);

						float baba=total_power/(r_power+g_power+b_power);

						r_power*=baba;
						g_power*=baba;
						b_power*=baba;
					}
				}

				if( type=='s' )
				{
					R3_vector view=(origin-intersection).normalized( );
					float mult=1.0/spheres[final_index].get_material( ).ref_index;
					float nDotv=(view).dot(hemisphere_direction);
					float sin2_ang=mult*mult*(1-nDotv*nDotv);
					direction=-mult*view+(mult*nDotv-sqrt(1-sin2_ang))*hemisphere_direction;
					direction=direction.normalized( );
					origin=intersection+0.01*direction;
					float param=find_intersection_parameter_sphere( origin, direction, spheres[final_index] );
					intersection=origin+param*direction;

					hemisphere_direction=spheres[final_index].get_center( )-intersection;
					hemisphere_direction=hemisphere_direction.normalized( );
					mult=spheres[final_index].get_material( ).ref_index;
					view=(origin-intersection).normalized( );
					nDotv=(view).dot(hemisphere_direction);
					sin2_ang=mult*mult*(1-nDotv*nDotv);
					direction=-mult*view+(mult*nDotv-sqrt(1-sin2_ang))*hemisphere_direction;
					direction=direction.normalized( );
					origin=intersection+0.01*direction;
				}
				if( type=='t' )
				{
					hemisphere_direction=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
								 cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
					hemisphere_direction=hemisphere_direction.normalized( );
					if( hemisphere_direction.dot( origin - triangles[final_index].get_vertex1( ) ) < 0 )
					{
						hemisphere_direction=(-1.0)*hemisphere_direction;
					}
					R3_vector view=(intersection-origin).normalized( );
					float mult=1.0/triangles[final_index].get_material( ).ref_index;
					float sin2_ang=pow(mult,2)*(1-pow(view.dot(hemisphere_direction),2));
					direction=mult*view+(mult*view.dot(hemisphere_direction)-sqrt(1-sin2_ang))*hemisphere_direction;
					direction=direction.normalized( );
					origin=intersection+0.01*direction;
				}

				t=-1;
				index=0;
				type='a';
				final_index=0;

				while( index < s_size )
				{
					t1=find_intersection_parameter_sphere( origin, direction, spheres[index] );
					if( t1 > 0 )
					{
						if( t < 0 )
						{
							t=t1;
							final_index=index;
							type='s';
						}
						else
						{
							if( t1 < t )
							{	
								t=t1;	
								final_index=index;
								type='s';
							}
						}
					}
					index++;
				}
				index=0;
				while( index < t_size )
				{
					t1=find_intersection_parameter_triangle( origin, direction, triangles[index] );
					if( t1 > 0 )
					{
						if( t < 0 )
						{
							t=t1;
							final_index=index;
							type='t';
						}
						else
						{
							if( t1 < t )
							{
								t=t1;
								final_index=index;
								type='t';
							}
						}
					}
					index++;
				}
				intersection=origin+t*direction;

				if( type=='t' )
				{
					hemisphere_direction=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
													cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );                

					hemisphere_direction=hemisphere_direction.normalized( );
					if( hemisphere_direction.dot( direction ) > 0 )
					{
						hemisphere_direction=(-1.0)*hemisphere_direction;
					}
					kd_avg=(triangles[final_index].get_material( ).k_dr+triangles[final_index].get_material( ).k_dg+
							triangles[final_index].get_material( ).k_db)/3.0;
					ks_avg=(triangles[final_index].get_material( ).k_sr+triangles[final_index].get_material( ).k_sg+
							triangles[final_index].get_material( ).k_sb)/3.0;
					kt_avg=(triangles[final_index].get_material( ).k_tr+triangles[final_index].get_material( ).k_tg+
							triangles[final_index].get_material( ).k_tb)/3.0;
				}
				if( type=='s' )
				{
					hemisphere_direction=intersection-spheres[final_index].get_center( );                

					hemisphere_direction=hemisphere_direction.normalized( );
					kd_avg=(spheres[final_index].get_material( ).k_dr+spheres[final_index].get_material( ).k_dg+
							spheres[final_index].get_material( ).k_db)/3.0;
					ks_avg=(spheres[final_index].get_material( ).k_sr+spheres[final_index].get_material( ).k_sg+
							spheres[final_index].get_material( ).k_sb)/3.0;
					kt_avg=(spheres[final_index].get_material( ).k_tr+spheres[final_index].get_material( ).k_tg+
							spheres[final_index].get_material( ).k_tb)/3.0;
				}
				is_indirect=0;
				is_direct=0;
				is_caust=1;
			}
			if( what < 5.0 )
			{
				what=1.0*rand( )/RAND_MAX;
			}
		}
		if( t!= -1 && kd_avg > 0 )
		{
			if( do_color_bleeding )
			{
				float mean=(r_power+g_power+b_power)/3.0;
				if( type=='s' )
				{	
					/* r_power=lights[light_index].get_total_power( )*std::min(spheres[final_index].get_material( ).k_dr,r_power/lights[light_index].get_total_power( ));
					g_power=lights[light_index].get_total_power( )*std::min(spheres[final_index].get_material( ).k_dg,g_power/lights[light_index].get_total_power( ));
					b_power=lights[light_index].get_total_power( )*std::min(spheres[final_index].get_material( ).k_db,b_power/lights[light_index].get_total_power( )); */

					/* r_power*=spheres[final_index].get_material( ).k_dr;
					g_power*=spheres[final_index].get_material( ).k_dg;
					b_power*=spheres[final_index].get_material( ).k_db; */

					r_power=.5*(r_power/mean+spheres[final_index].get_material( ).k_dr);
					g_power=.5*(g_power/mean+spheres[final_index].get_material( ).k_dg);
					b_power=.5*(b_power/mean+spheres[final_index].get_material( ).k_db);

					float baba=total_power/(r_power+g_power+b_power);

					r_power*=baba;
					g_power*=baba;
					b_power*=baba;

					/* r_power=(r_power+r_power*spheres[final_index].get_material( ).k_dr)/2;
					g_power=(g_power+g_power*spheres[final_index].get_material( ).k_dg)/2;
					b_power=(b_power+b_power*spheres[final_index].get_material( ).k_db)/2; */
				}
				if( type=='t' )
				{	
					/* r_power=lights[light_index].get_total_power( )*std::min(triangles[final_index].get_material( ).k_dr,r_power/lights[light_index].get_total_power( ));
					g_power=lights[light_index].get_total_power( )*std::min(triangles[final_index].get_material( ).k_dg,g_power/lights[light_index].get_total_power( ));
					b_power=lights[light_index].get_total_power( )*std::min(triangles[final_index].get_material( ).k_db,b_power/lights[light_index].get_total_power( )); */

					/* r_power*=triangles[final_index].get_material( ).k_dr;
					g_power*=triangles[final_index].get_material( ).k_dg;
					b_power*=triangles[final_index].get_material( ).k_db; */

					r_power=.5*(r_power/mean+triangles[final_index].get_material( ).k_dr);
					g_power=.5*(g_power/mean+triangles[final_index].get_material( ).k_dg);
					b_power=.5*(b_power/mean+triangles[final_index].get_material( ).k_db);

					float baba=total_power/(r_power+g_power+b_power);

					r_power*=baba;
					g_power*=baba;
					b_power*=baba;

					/* r_power=(r_power+r_power*triangles[final_index].get_material( ).k_dr)/2;
					g_power=(g_power+g_power*triangles[final_index].get_material( ).k_dg)/2;
					b_power=(b_power+b_power*triangles[final_index].get_material( ).k_db)/2; */
				}
			}
			if( !one_map )
			{
				if( is_direct && t!=-1 && do_direct )
				{
					direct_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
					is_direct=0;
					t=-1;
				}
				if( is_indirect && t!=-1 && do_indirect )
				{
					indirect_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
					is_indirect=0;
					what=5.0;
					t=-1;
				}
				if( is_caust && t!=-1 && do_caust )
				{
					caust_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
					is_caust=0;
					what=5.0;
					t=-1;
				}
			}
			else
			{
				if( emissive_photons < total_photons )
				{
					direct_photons.push_back( new Photon( intersection, r_power, g_power, b_power, direction ) );
				}
			}
		}
	}

	int number_of_photons_d=direct_photons.size( );
	int num_indirect_photons=indirect_photons.size( );
	int num_caust_photons=caust_photons.size( );

	for( int i=0; i < number_of_photons_d; i++ )
	{
		direct_photons[i]->scale_r_power( 1.0/emissive_photons );
		direct_photons[i]->scale_g_power( 1.0/emissive_photons );
		direct_photons[i]->scale_b_power( 1.0/emissive_photons );
	}
	for( int i=0; i < num_indirect_photons; i++ )
	{
		indirect_photons[i]->scale_r_power( 1.0/emissive_photons );
		indirect_photons[i]->scale_g_power( 1.0/emissive_photons );
		indirect_photons[i]->scale_b_power( 1.0/emissive_photons );
	}
	for( int i=0; i < num_caust_photons; i++ )
	{
		caust_photons[i]->scale_r_power( 1.0/emissive_photons );
		caust_photons[i]->scale_g_power( 1.0/emissive_photons );
		caust_photons[i]->scale_b_power( 1.0/emissive_photons );
	}
}

int pmf::partition( std::vector<Photon*>& photons, Photon* pho, int p, int r, short& axis )
{
	int index=0;
	for( int i=p; i<=r; i++ )
	{
		if( photons[i]==pho )
		{
			index=i;
			break;
		}
	}
	Photon* temp=photons[r];
	photons[r]=photons[index];
	photons[index]=temp;

	/* for( int i=0; i<photons.size(); i++ )
	{
		std::cout << photons[i]->get_position() << "\n";
	}
	std::cout << "\n"; */

	int i=p-1;
	if ( axis==0 )
	{
		for( int j=p; j<r; j++ )
		{
			if( photons[j]->get_position( ).get_x_comp( ) <= photons[r]->get_position( ).get_x_comp( ) )
			{
				i+=1;
				temp=photons[i];
				photons[i]=photons[j];
				photons[j]=temp;
			}
		}
	}
	if ( axis==1 )
	{
		for( int j=p; j<r; j++ )
		{
			if( photons[j]->get_position( ).get_y_comp( ) <= photons[r]->get_position( ).get_y_comp( ) )
			{
				i+=1;
				temp=photons[i];
				photons[i]=photons[j];
				photons[j]=temp;
			}
		}
	}
	if ( axis==2 )
	{
		for( int j=p; j<r; j++ )
		{
			if( photons[j]->get_position( ).get_z_comp( ) <= photons[r]->get_position( ).get_z_comp( ) )
			{
				i+=1;
				temp=photons[i];
				photons[i]=photons[j];
				photons[j]=temp;
			}
		}
	}
	temp=photons[i+1];
	photons[i+1]=photons[r];
	photons[r]=temp;
	return i+1;
}

Photon* pmf::get_median( std::vector<Photon*>& photons, int p, int r, int m, short& axis )
{
	if( p==r )
	{
		// std::cout << photons[p]->get_position() << "\n\n";
		return photons[p];
	}
	
	int part=rand( )%(r-p+1)+p;
	int q=partition( photons, photons[part], p, r, axis );
	// Photon* part=get_median_sub( photons, axis );

	int k=q-p;

	if( k==m )
	{
		// std::cout << photons[q]->get_position() << "\n\n";
		return photons[q];
	}
	else
	{
		if( m < k )
		{
			return get_median( photons, p, q-1, m, axis );
		}
		else
		{
			return get_median( photons, q+1, r, m-k-1, axis );
		}
	}
}

Photon* pmf::get_median_large( std::vector<Photon*>& photons, int p, int r, int m, short& axis )
{
	bool median=false;
	while( !median )
	{
		if( p==r )
		{
			// std::cout << photons[p]->get_position() << "\n\n";
			return photons[p];
		}
	
		int part=rand( )%(r-p+1)+p;
		int q=partition( photons, photons[part], p, r, axis );
		// Photon* part=get_median_sub( photons, axis );

		int k=q-p;

		if( k==m )
		{
			// std::cout << photons[q]->get_position() << "\n\n";
			return photons[q];
		}
		else
		{
			if( m < k )
			{
				r=q-1;
				// return get_median( photons, p, q-1, m, axis );
			}
			else
			{
				p=q+1;
				m=m-k-1;
				// return get_median( photons, q+1, r, m-k-1, axis );
			}
		}
	}
}

void pmf::build_photon_map( std::vector<Photon*>& kd_tree, std::vector<Photon*>& photons, int n )
{
	int p_size=photons.size( );
	float x_max=photons[0]->get_position( ).get_x_comp( );
	float x_min=photons[0]->get_position( ).get_x_comp( ); 
	float y_max=photons[0]->get_position( ).get_y_comp( );  
	float y_min=photons[0]->get_position( ).get_y_comp( ); 
	float z_max=photons[0]->get_position( ).get_z_comp( );  
	float z_min=photons[0]->get_position( ).get_z_comp( );
	for( int i=1; i<p_size; i++ )
	{
		if( photons[i]->get_position( ).get_x_comp( ) < x_min )
		{
			x_min=photons[i]->get_position( ).get_x_comp( );
		}
		if( photons[i]->get_position( ).get_x_comp( ) > x_max )
		{
			x_max=photons[i]->get_position( ).get_x_comp( );
		}
		if( photons[i]->get_position( ).get_y_comp( ) < y_min )
		{
			y_min=photons[i]->get_position( ).get_y_comp( );
		}
		if( photons[i]->get_position( ).get_y_comp( ) > y_max )
		{
			y_max=photons[i]->get_position( ).get_y_comp( );
		}
		if( photons[i]->get_position( ).get_z_comp( ) < z_min )
		{
			z_min=photons[i]->get_position( ).get_z_comp( );
		}
		if( photons[i]->get_position( ).get_z_comp( ) > z_max )
		{
			z_max=photons[i]->get_position( ).get_z_comp( );
		}
	}
	short split=-1;
	x_max-=x_min;
	y_max-=y_min;
	z_max-=z_min;
	if( y_max <= x_max && z_max <= x_max)
	{
		split=0;
	}
	if( x_max <= y_max && z_max <= y_max)
	{
		split=1;
	}
	if( x_max <= z_max && y_max <= z_max)
	{
		split=2;
	}

	/* std::cout << "split: " << split << "\n";
	for( int i=0; i<p_size; i++ )
	{
		std::cout << photons[i]->get_position() << "\n";
	}
	std::cout << "\n"; */

	kd_tree[n]=get_median( photons, 0, p_size-1, ceil(1.0*p_size/2)-1, split );
	kd_tree[n]->set_plane(split);
	// std::cout << "\n\nn: " << n << "\n\n";
	// std::cout << "median: " << kd_tree[n]->get_position() << "\n";
	// int z=partition(photons, kd_tree[n], 0, p_size-1, split );

	/* for( int i=0; i<p_size; i++ )
	{
		std::cout << photons[i]->get_position() << "\n";
	}
	std::cout << "\n"; */
	
	std::vector<Photon*> left(ceil(1.0*p_size/2)-1), right(int(1.0*p_size/2));
	for( int i=0; i<ceil(1.0*p_size/2)-1; i++ )
	{
		left[i]=photons[i];
	}
	for( int i=ceil(1.0*p_size/2); i<p_size; i++ )
	{
		right[i-ceil(1.0*p_size/2)]=photons[i];
	}

	/* std::cout << "Left:\n";
	for( int i=0; i<left.size(); i++ )
	{
		std::cout << left[i]->get_position() << "\n";
	}
	std::cout << "\n";

	std::cout << "Right:\n";
	for( int i=0; i<right.size(); i++ )
	{
		std::cout << right[i]->get_position() << "\n";
	}
	std::cout << "\n"; */

	if( unsigned(2*(n+1)) <= kd_tree.size( ) )
	{
		build_photon_map( kd_tree, left, 2*(n+1)-1 );
	}
	if( unsigned(2*(n+1)+1) <= kd_tree.size( ) )
	{
		build_photon_map( kd_tree, right, 2*(n+1) );
	}
}

void pmf::build_photon_map_large( std::vector<Photon*>& kd_tree, std::vector<Photon*>& photons, int n, int p )
{
	// std::vector< std::vector<Photon*> > stack;
	std::vector<int> indices, n_indices, p_indices;
	// stack.push_back( photons );
	indices.push_back( n );
	n_indices.push_back( n );
	p_indices.push_back( p );
	int kd_size=kd_tree.size( );

	while( !indices.empty( ) )
	{
		// int p_size=stack.back( ).size( );
		float x_max=photons[n_indices.back( )]->get_position( ).get_x_comp( );
		float x_min=photons[n_indices.back( )]->get_position( ).get_x_comp( ); 
		float y_max=photons[n_indices.back( )]->get_position( ).get_y_comp( );  
		float y_min=photons[n_indices.back( )]->get_position( ).get_y_comp( ); 
		float z_max=photons[n_indices.back( )]->get_position( ).get_z_comp( );  
		float z_min=photons[n_indices.back( )]->get_position( ).get_z_comp( );
		for( int i=n_indices.back( )+1; i<=p_indices.back( ); i++ )
		{
			if( photons[i]->get_position( ).get_x_comp( ) < x_min )
			{
				x_min=photons[i]->get_position( ).get_x_comp( );
			}
			if( photons[i]->get_position( ).get_x_comp( ) > x_max )
			{
				x_max=photons[i]->get_position( ).get_x_comp( );
			}
			if( photons[i]->get_position( ).get_y_comp( ) < y_min )
			{
				y_min=photons[i]->get_position( ).get_y_comp( );
			}
			if( photons[i]->get_position( ).get_y_comp( ) > y_max )
			{
				y_max=photons[i]->get_position( ).get_y_comp( );
			}
			if( photons[i]->get_position( ).get_z_comp( ) < z_min )
			{
				z_min=photons[i]->get_position( ).get_z_comp( );
			}
			if( photons[i]->get_position( ).get_z_comp( ) > z_max )
			{
				z_max=photons[i]->get_position( ).get_z_comp( );
			}
		}
		short split=-1;
		x_max-=x_min;
		y_max-=y_min;
		z_max-=z_min;
		if( y_max <= x_max && z_max <= x_max)
		{
			split=0;
		}
		if( x_max <= y_max && z_max <= y_max)
		{
			split=1;
		}
		if( x_max <= z_max && y_max <= z_max)
		{
			split=2;
		}

		/* std::cout << "split: " << split << "\n";
		for( int i=0; i<p_size; i++ )
		{
			std::cout << photons[i]->get_position() << "\n";
		}
		std::cout << "\n"; */

		kd_tree[indices.back( )]=get_median_large( photons, n_indices.back( ), p_indices.back( ), 
												   ceil(1.0*(p_indices.back( )-n_indices.back( ))/2), split );
		kd_tree[indices.back( )]->set_plane(split);
		// std::cout << "\n\nn: " << n << "\n\n";
		// std::cout << "median: " << kd_tree[n]->get_position() << "\n";
		// int z=partition(photons, kd_tree[n], 0, p_size-1, split );

		/* for( int i=0; i<p_size; i++ )
		{
			std::cout << photons[i]->get_position() << "\n";
		}
		std::cout << "\n"; */
	
		/* std::vector<Photon*> left(ceil(1.0*p_size/2)-1), right(int(1.0*p_size/2));
		for( int i=0; i<ceil(1.0*p_size/2)-1; i++ )
		{
			left[i]=stack.back( )[i];
		}
		for( int i=ceil(1.0*p_size/2); i<p_size; i++ )
		{
			right[i-ceil(1.0*p_size/2)]=stack.back( )[i];
		} */

		/* std::cout << "Left:\n";
		for( int i=0; i<left.size(); i++ )
		{
			std::cout << left[i]->get_position() << "\n";
		}
		std::cout << "\n";

		std::cout << "Right:\n";
		for( int i=0; i<right.size(); i++ )
		{
			std::cout << right[i]->get_position() << "\n";
		}
		std::cout << "\n"; */

		// stack.pop_back( );
		int left_index=2*(indices.back( )+1)-1;
		int left_n=p_indices.back( )-2.0*ceil(1.0*(p_indices.back( )-n_indices.back( ))/2);
		int right_index=2*(indices.back( )+1);
		int right_n=n_indices.back( )+1.0*ceil(1.0*(p_indices.back( )-n_indices.back( ))/2)+1;
		int right_p=right_n+1.0*ceil(1.0*(p_indices.back( )-n_indices.back( ))/2)-1;
		indices.pop_back( );
		n_indices.pop_back( );
		p_indices.pop_back( );

		if( left_index+1 <= kd_size )
		{
			// stack.push_back( left );
			indices.push_back( left_index );
			n_indices.push_back( left_n );
			p_indices.push_back( right_n-2 );
		}
		if( right_index+1 <= kd_size )
		{
			// stack.push_back( right );
			indices.push_back( right_index );
			n_indices.push_back( right_n );
			p_indices.push_back( right_p );
		}
	}
}

void pmf::illumination( hs::RGB& pix, R3_vector e, R3_vector cop_direction, std::vector<Sphere>& spheres, std::vector<Triangle>& triangles, 
						R3_vector lloc, ANNkd_tree* kd_tree_direct, Point* points_direct, ANNkd_tree* kd_tree_indirect, Point* points_indirect,
						ANNkd_tree* kd_tree_caust, Point* points_caust, float r_scale, float g_scale, float b_scale, int num_photons_d, 
						int num_photons_i, int num_photons_c, bool do_global, bool direct, Light_source light, int light_samples,
						int& recursion_depth, bool do_cone_filter, bool do_attenuation, bool do_direct, bool do_indirect, 
						bool do_caust, bool do_cone_filter_all )
						
						// std::vector<Photon*>& kd_tree, float r_scale, float g_scale, 
						// float b_scale )
{
	if( (r_scale > 0.001 || g_scale > 0.001 || b_scale > 0.001) && recursion_depth < 12 )
	{
		float c0=1, c1=1.0/light.get_area( ), c2=1.0/(light.get_area( )*light.get_area( ));
		// float c0=0, c1=0, c2=1;
		int s_size=spheres.size( );
		int t_size=triangles.size( );
		// int l_size=lights.size( );
		int index=0;
		char type='a';
		int final_index=0;

		float t=-1;
		float t1;
		R3_vector intersection;
		R3_vector temp_intersection;
		// for( int j=0; j<l_size; j++ )
		{
			while( index < s_size )
			{
				t1=find_intersection_parameter_sphere( e, cop_direction, spheres[index] );
				if( t1 > 0 )
				{
					if( t < 0 )
					{
						t=t1;
						final_index=index;
						type='s';
					}
					else
					{
						if( t1 < t )
						{	
							t=t1;	
							final_index=index;
							type='s';
						}
					}
				}
				index++;
			}
			index=0;
			while( index < t_size )
			{
				t1=find_intersection_parameter_triangle( e, cop_direction, triangles[index] );
				if( t1 > 0 )
				{
					if( t < 0 )
					{
						t=t1;
						final_index=index;
						type='t';
					}
					else
					{
						if( t1 < t )
						{
							t=t1;
							final_index=index;
							type='t';
						}
					}
				}
				index++;
			}
			if( type == 's' )
			{
				intersection=e+t*cop_direction;
				R3_vector temp_intersection=intersection, cop_temp=cop_direction, e_temp=e, N;
				if( spheres[final_index].get_material( ).type==1 )
				{
					pix.r+=(1.0*r_scale);
					pix.g+=(1.0*g_scale);
					pix.b+=(1.0*b_scale);
				}
				if( spheres[final_index].get_material( ).type==2 )
				{
					R3_vector N=intersection-spheres[final_index].get_center( );
					N=N.normalized( );
					t=-1;
					if(  N.dot( lloc-intersection ) >= 0 )
					{
						R3_vector d=lloc-intersection;
						int i=0;
						while( i < s_size && (t >= 1 || t <= 0) )
						{
							if( i != final_index )
							{
								t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
							}
							i++;				
						}
						i=0;
						while( i < t_size && (t >= 1 || t <= 0) )
						{
							t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
							i++;
						}
						if( t >= 1 || t <= 0 )
						{
							R3_vector L=lloc-intersection;
							float mult;
							if( do_attenuation )
							{
								float len=L.norm( );
								mult=1.0/(c0+c1*len+c2*len*len);
							}
							else
							{
								mult=1.0;
							}
							L=L.normalized( );
							R3_vector V=e-intersection;
							V=V.normalized( );
							R3_vector H=L+V;
							H=H.normalized( );
								float n=0.5*snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))*
											sin(8/exp(snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))))+0.5;
							vec3 C=vec3(0.5*snoise(vec3(s*0.5*intersection.get_x_comp( ), s*0.5*intersection.get_y_comp( ), s*0.5*intersection.get_z_comp( )))+
											0.5,0.5*snoise(vec3(s*intersection.get_x_comp( ), s*intersection.get_y_comp( ), s*intersection.get_z_comp( )))+0.5,
											0.5*snoise(vec3(3.0/2.0*intersection.get_x_comp( ), s*3.0/2.0*intersection.get_y_comp( ), s*3.0/2.0*intersection.get_z_comp( )))+0.5);

								// vec3 color=n*((C*(N.dot(L)))*lights[j].get_intensity( ));
							vec3 color=n*((C*(N.dot(L))));

							pix.r += mult*r_scale*color[0]*light.get_intensity( );
							pix.g += mult*g_scale*color[1]*light.get_intensity( );
							pix.b += mult*b_scale*color[2]*light.get_intensity( );
						}
					}
					if( do_global )
					{
						if( do_direct )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_direct, points_direct,
								r_scale, g_scale, b_scale, num_photons_d, 0, 0 );
						}
						if( do_indirect )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_indirect, points_indirect,
								r_scale, g_scale, b_scale, num_photons_i, 0, 0 );
						}
						if( do_caust )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_caust, points_caust,
								r_scale, g_scale, b_scale, num_photons_c, do_cone_filter, 1 );
						}
					}
				}
				if( spheres[final_index].get_material( ).k_tr>0 || spheres[final_index].get_material( ).k_tg>0 ||
					spheres[final_index].get_material( ).k_tb>0 )
				{
					intersection=e+t*cop_direction;
					R3_vector temp_intersection=intersection, cop_temp=cop_direction, e_temp=e, N, temp_intersection2, cop_temp2, e_temp2, view;
					float r_scale2, g_scale2, b_scale2;
					float r_scale3, g_scale3, b_scale3;
					float mult, sin2_ang, param;
					bool test=0;
					N=intersection-spheres[final_index].get_center( );
					if( (e-spheres[final_index].get_center( )).norm( ) > N.norm( ) )
					{
						N=N.normalized( );
						view=(e-intersection).normalized( );
						r_scale2=pow((1-spheres[final_index].get_material( ).ref_index)/(1+spheres[final_index].get_material( ).ref_index),2);
						r_scale2+=((1-r_scale2)*pow(1-N.dot(view),5));
						r_scale2=r_scale2*r_scale*spheres[final_index].get_material( ).k_tr;
						g_scale2=pow((1-spheres[final_index].get_material( ).ref_index)/(1+spheres[final_index].get_material( ).ref_index),2);
						g_scale2+=((1-g_scale2)*pow(1-N.dot(view),5));
						g_scale2=g_scale2*g_scale*spheres[final_index].get_material( ).k_tg;
						b_scale2=pow((1-spheres[final_index].get_material( ).ref_index)/(1+spheres[final_index].get_material( ).ref_index),2);
						b_scale2+=((1-b_scale2)*pow(1-N.dot(view),5));
						b_scale2=b_scale2*b_scale*spheres[final_index].get_material( ).k_tb;
						mult=1.0/spheres[final_index].get_material( ).ref_index;
						sin2_ang=pow(mult,2)*(1-pow(view.dot(N),2));
						cop_direction=-mult*view+(mult*view.dot(N)-sqrt(1-sin2_ang))*N;
						cop_direction=cop_direction.normalized( );
						e=intersection+0.01*cop_direction;
						param=find_intersection_parameter_sphere( e, cop_direction, spheres[final_index] );
						intersection=e+param*cop_direction;
						recursion_depth++;
						illumination( pix, e, cop_direction, spheres, triangles, lloc, kd_tree_direct, points_direct, kd_tree_indirect, points_indirect, 
							kd_tree_caust, points_caust, r_scale-r_scale2, g_scale-g_scale2, b_scale-b_scale2, num_photons_d, 
							num_photons_i, num_photons_c, do_global, direct, light, light_samples, recursion_depth, do_cone_filter,
							do_attenuation, do_direct, do_indirect, do_caust, do_cone_filter_all );
						recursion_depth--;

						temp_intersection2=intersection;
						cop_temp2=cop_direction;
						e_temp2=e;
						test=1;
					}
					else
					{
						r_scale2=r_scale; 
						g_scale2=g_scale;
						b_scale2=b_scale;
						temp_intersection2=temp_intersection;
						cop_temp2=cop_temp;
						e_temp2=e_temp;
					}

					N=intersection-spheres[final_index].get_center( );
					mult=spheres[final_index].get_material( ).ref_index;
					view=(e-intersection).normalized( );
					if( view.dot(N) < 0 )
					{
						N=-1.0*N.normalized( );
					}
					r_scale3=pow((spheres[final_index].get_material( ).ref_index-1)/(1+spheres[final_index].get_material( ).ref_index),2);
					r_scale3+=((1-r_scale3)*pow(1-N.dot(view),5));
					r_scale3=r_scale2*r_scale3*spheres[final_index].get_material( ).k_tr;
					g_scale3=pow((spheres[final_index].get_material( ).ref_index-1)/(1+spheres[final_index].get_material( ).ref_index),2);
					g_scale3+=((1-g_scale3)*pow(1-N.dot(view),5));
					g_scale3=g_scale2*g_scale3*spheres[final_index].get_material( ).k_tg;
					b_scale3=pow((spheres[final_index].get_material( ).ref_index-1)/(1+spheres[final_index].get_material( ).ref_index),2);
					b_scale3+=((1-b_scale3)*pow(1-N.dot(view),5));
					b_scale3=b_scale2*b_scale3*spheres[final_index].get_material( ).k_tb;
					float nDotv=view.dot(N);
					sin2_ang=pow(mult,2)*(1-pow(nDotv,2));
					cop_direction=-mult*view+(mult*nDotv-sqrt(1-sin2_ang))*N;
					cop_direction=cop_direction.normalized( );
					e=intersection+0.01*cop_direction;

					recursion_depth++;
					illumination( pix, e, cop_direction, spheres, triangles, lloc, kd_tree_direct, points_direct, kd_tree_indirect, points_indirect, 
						kd_tree_caust, points_caust, r_scale2-r_scale3, g_scale2-g_scale3, b_scale2-b_scale3, 
						num_photons_d, num_photons_i, num_photons_c, do_global, direct, light, light_samples, recursion_depth, do_cone_filter,
						do_attenuation, do_direct, do_indirect, do_caust, do_cone_filter_all );
					recursion_depth--;
					intersection=temp_intersection2;
					e=e_temp2;
					cop_direction=cop_temp2;
					cop_direction=(intersection-e).normalized( )-(2*((intersection-e).normalized( )).
						dot((spheres[final_index].get_center( )-intersection).normalized( )))*
						(spheres[final_index].get_center( )-intersection).normalized( );
					e=intersection+.01*cop_direction;
					recursion_depth++;
					illumination( pix, e, cop_direction, spheres, triangles, lloc, kd_tree_direct, points_direct, kd_tree_indirect, points_indirect,
						kd_tree_caust, points_caust, r_scale3, g_scale3, b_scale3, num_photons_d, num_photons_i, num_photons_c,
						do_global, direct, light, light_samples, recursion_depth, do_cone_filter, do_attenuation, do_direct,
						do_indirect, do_caust, do_cone_filter_all );
					recursion_depth--;
					if( test )
					{
						intersection=temp_intersection;
						e=e_temp;
						cop_direction=cop_temp;
						cop_direction=(intersection-e).normalized( )-(2*((intersection-e).normalized( )).
							dot((intersection-spheres[final_index].get_center( )).normalized( )))*
							(intersection-spheres[final_index].get_center( )).normalized( );
						e=intersection+.01*cop_direction;
						recursion_depth++;
						illumination( pix, e, cop_direction, spheres, triangles, lloc, kd_tree_direct, points_direct, kd_tree_indirect, points_indirect,
							kd_tree_caust, points_caust, r_scale2, g_scale2, b_scale2, num_photons_d, num_photons_i, num_photons_c,
							do_global, direct, light, light_samples, recursion_depth, do_cone_filter, do_attenuation, do_direct, 
							do_indirect, do_caust, do_cone_filter_all );
						recursion_depth--;
					}
				}
				if( spheres[final_index].get_material( ).k_sr>0.0 || spheres[final_index].get_material( ).k_sg>0.0 ||
					spheres[final_index].get_material( ).k_sb>0.0 )
				{
					cop_direction=(intersection-e).normalized( )-(2*((intersection-e).normalized( )).
						dot((intersection-spheres[final_index].get_center( )).normalized( )))*
						(intersection-spheres[final_index].get_center( )).normalized( );
					e=intersection+.01*cop_direction;
					recursion_depth++;
					illumination( pix, e, cop_direction, spheres, triangles, lloc, kd_tree_direct, points_direct, kd_tree_indirect, points_indirect, 
						kd_tree_caust, points_caust, r_scale*spheres[final_index].get_material( ).k_sr, 
						g_scale*spheres[final_index].get_material( ).k_sg, b_scale*spheres[final_index].get_material( ).k_sb, 
						num_photons_d, num_photons_i, num_photons_c, do_global, direct, light, light_samples, recursion_depth, do_cone_filter,
						do_attenuation, do_direct, do_indirect, do_caust, do_cone_filter_all );
					recursion_depth--;
				}
				if( spheres[final_index].get_material( ).k_dr>0.0 || spheres[final_index].get_material( ).k_dg>0.0 ||
					spheres[final_index].get_material( ).k_db>0.0 )
				{
					if( direct )
					{
						int shadows=0;
						if( light_samples > 1 )
						{
							for( int i=0; i<4; i++ )
							{
								switch( i )
								{
									case 0: lloc=light.get_ll_corner( );
										break;
									case 1: lloc=light.get_lr_corner( );
										break;
									case 2: lloc=light.get_ul_corner( );
										break;
									case 3: lloc=light.get_ur_corner( );
										break;
								}
								R3_vector N=intersection-spheres[final_index].get_center( );
								N=N.normalized( );
								t=-1;
								if(  N.dot( lloc-intersection ) >= 0 )
								{
									R3_vector d=lloc-intersection;
									int i=0;
									while( i < s_size && (t >= 1 || t <= 0) )
									{
										if( i != final_index )
										{
											t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
										}
										i++;				
									}
									i=0;
									while( i < t_size && (t >= 1 || t <= 0) )
									{
										t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
										i++;
									}
									if( t <= 1 && t >= 0 )
									{
										shadows++;
									}
								}
							}
							if( shadows > 0 && shadows < 4 )
							{
								float r=0, g=0, b=0;
								for( int i=0; i<light_samples; i++ )
								{
									lloc=light.get_location( );
									R3_vector N=intersection-spheres[final_index].get_center( );
									N=N.normalized( );
									t=-1;
									if(  N.dot( lloc-intersection ) >= 0 )
									{
										R3_vector d=lloc-intersection;
										int i=0;
										while( i < s_size && (t >= 1 || t <= 0) )
										{
											if( i != final_index )
											{
												t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
											}
											i++;				
										}
										i=0;
										while( i < t_size && (t >= 1 || t <= 0) )
										{
											t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
											i++;
										}
										if( t >= 1 || t <= 0 )
										{
											R3_vector L=lloc-intersection;
											float mult;
											if( do_attenuation )
											{
												float len=L.norm( );
												mult=1.0/(c0+c1*len+c2*len*len);
											}
											else
											{
												mult=1.0;
											}
											L=L.normalized( );
											R3_vector V=e-intersection;
											V=V.normalized( );
											R3_vector H=L+V;
											H=H.normalized( );
											r += r_scale*mult*
												((spheres[final_index].get_material( ).k_dr)*(N.dot( L ))+
												(spheres[final_index].get_material( ).k_sr)*pow(H.dot( N ), 
												spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
											g += g_scale*mult*
												((spheres[final_index].get_material( ).k_dg)*(N.dot( L ))+
												(spheres[final_index].get_material( ).k_sg)*pow(H.dot( N ), 
												spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
											b += b_scale*mult*
												((spheres[final_index].get_material( ).k_db)*(N.dot( L ))+
												(spheres[final_index].get_material( ).k_sb)*pow(H.dot( N ), 
												spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
										}
									}
								}
								pix.r+=(r/light_samples);
								pix.g+=(g/light_samples);
								pix.b+=(b/light_samples);
							}
							if( shadows==0 )
							{
								lloc=light.get_center( );
								R3_vector N=intersection-spheres[final_index].get_center( );
								N=N.normalized( );
								t=-1;
								if(  N.dot( lloc-intersection ) >= 0 )
								{
									R3_vector d=lloc-intersection;
									int i=0;
									while( i < s_size && (t >= 1 || t <= 0) )
									{
										if( i != final_index )
										{
											t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
										}
										i++;				
									}
									i=0;
									while( i < t_size && (t >= 1 || t <= 0) )
									{
										t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
										i++;
									}
									if( t >= 1 || t <= 0 )
									{
										R3_vector L=lloc-intersection;
										float mult;
										if( do_attenuation )
										{
											float len=L.norm( );
											mult=1.0/(1+c1*len+c2*len*len);
										}
										else
										{
											mult=1.0;
										}
										L=L.normalized( );
										R3_vector V=e-intersection;
										V=V.normalized( );
										R3_vector H=L+V;
										H=H.normalized( );
										pix.r += r_scale*mult*
											((spheres[final_index].get_material( ).k_dr)*(N.dot( L ))+
											(spheres[final_index].get_material( ).k_sr)*pow(H.dot( N ), 
											spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
										pix.g += g_scale*mult*
											((spheres[final_index].get_material( ).k_dg)*(N.dot( L ))+
											(spheres[final_index].get_material( ).k_sg)*pow(H.dot( N ), 
											spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
										pix.b += b_scale*mult*
											((spheres[final_index].get_material( ).k_db)*(N.dot( L ))+
											(spheres[final_index].get_material( ).k_sb)*pow(H.dot( N ), 
											spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
									}
								}
							}
						}
						else
						{
							R3_vector N=intersection-spheres[final_index].get_center( );
							N=N.normalized( );
							t=-1;
							if(  N.dot( lloc-intersection ) >= 0 )
							{
								R3_vector d=lloc-intersection;
								int i=0;
								while( i < s_size && (t >= 1 || t <= 0) )
								{
									if( i != final_index )
									{
										t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
									}
									i++;				
								}
								i=0;
								while( i < t_size && (t >= 1 || t <= 0) )
								{
									t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
									i++;
								}
								if( t >= 1 || t <= 0 )
								{
									R3_vector L=lloc-intersection;
									float mult;
									if( do_attenuation )
									{
										float len=L.norm( );
										mult=1.0/(c0+c1*len+c2*len*len);
									}
									else
									{
										mult=1.0;
									}
									L=L.normalized( );
									R3_vector V=e-intersection;
									V=V.normalized( );
									R3_vector H=L+V;
									H=H.normalized( );
									pix.r += r_scale*mult*
										((spheres[final_index].get_material( ).k_dr)*(N.dot( L ))+
										(spheres[final_index].get_material( ).k_sr)*pow(H.dot( N ), 
										spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
									pix.g += g_scale*mult*
										((spheres[final_index].get_material( ).k_dg)*(N.dot( L ))+
										(spheres[final_index].get_material( ).k_sg)*pow(H.dot( N ), 
										spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
									pix.b += b_scale*mult*
										((spheres[final_index].get_material( ).k_db)*(N.dot( L ))+
										(spheres[final_index].get_material( ).k_sb)*pow(H.dot( N ), 
										spheres[final_index].get_material( ).n_spec))*light.get_intensity( );
								}
							}
						}
					}
					if( do_global )
					{
						if( do_direct )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_direct, points_direct,
								r_scale, g_scale, b_scale, num_photons_d, 0, 0 );
						}
						if( do_indirect )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_indirect, points_indirect,
								r_scale, g_scale, b_scale, num_photons_i, 0, 0 );
						}
						if( do_caust )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_caust, points_caust,
								r_scale, g_scale, b_scale, num_photons_c, do_cone_filter, 1 );
						}
					}
				}
			}
			if( type == 't' )
			{
				intersection=e+t*cop_direction;
				if( triangles[final_index].get_material( ).type==1 )
				{
					pix.r+=(light.get_intensity( )*r_scale);
					pix.g+=(1.0*g_scale*light.get_intensity( ));
					pix.b+=(1.0*b_scale*light.get_intensity( ));
				}
				if( triangles[final_index].get_material( ).type==2 )
				{
					R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
						cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
					t=-1;                

					N=N.normalized( );
					if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
					{
						N=(-1.0)*N;
					}
					if(  N.dot( lloc-intersection ) >= 0 )
					{
						R3_vector d=lloc-intersection;
						int i=0;
						while( i < t_size && (t >= 1 || t <= 0) )
						{
							if( i != final_index )
							{
								t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
							}
							i++;
						}
						i=0;
						while( i < s_size && (t >= 1 || t <= 0) )
						{
							t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
							i++;
						}
						if( t >= 1 || t <= 0 )
						{
							R3_vector L=lloc-intersection;
							float mult;
							if( do_attenuation )
							{
								float len=L.norm( );
								mult=1.0/(c0+c1*len+c2*len*len);
							}
							else
							{
								mult=1.0;
							}
							L=L.normalized( );
							R3_vector V=e-intersection;
							V=V.normalized( );
							R3_vector H=L+V;
							H=H.normalized( );

							float n=0.5*snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))*
								sin(8.0/exp(snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))))+0.5;
							vec3 C=vec3(0.5*snoise(vec3(s*0.5*intersection.get_x_comp( ), s*0.5*intersection.get_y_comp( ), s*0.5*intersection.get_z_comp( )))+0.5,
								0.5*snoise(vec3(s*intersection.get_x_comp( ), s*intersection.get_y_comp( ), s*intersection.get_z_comp( )))+0.5,
								0.5*snoise(vec3(s*3.0/2.0*intersection.get_x_comp( ), s*3.0/2.0*intersection.get_y_comp( ), s*3.0/2.0*intersection.get_z_comp( )))+0.5);

							vec3 color=n*((C*N.dot(L)));
							// vec3 color=((C*(N.dot(L)))*lights[j].get_intensity( ));

							pix.r += r_scale*mult*color[0];
							pix.g += g_scale*mult*color[1];
							pix.b += b_scale*mult*color[2];
						}
					}
					if( do_global )
					{
						if( do_direct )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_direct, points_direct,
								r_scale, g_scale, b_scale, num_photons_d, do_cone_filter_all, 0 );
						}
						if( do_indirect )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_indirect, points_indirect,
								r_scale, g_scale, b_scale, num_photons_i, do_cone_filter_all, 0 );
						}
						if( do_caust )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_caust, points_caust,
								r_scale, g_scale, b_scale, num_photons_c, do_cone_filter, 1 );
						}
					}
				}
				if( triangles[final_index].get_material( ).type==3 )
				{
					R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
						cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
					t=-1;                

					N=N.normalized( );
					if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
					{
						N=(-1.0)*N;
					}
					if(  N.dot( lloc-intersection ) >= 0 )
					{
						R3_vector d=lloc-intersection;
						int i=0;
						while( i < t_size && (t >= 1 || t <= 0) )
						{
							if( i != final_index )
							{
								t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
							}
							i++;
						}
						i=0;
						while( i < s_size && (t >= 1 || t <= 0) )
						{
							t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
							i++;
						}
						if( t >= 1 || t <= 0 )
						{
							R3_vector L=lloc-intersection;
							float mult;
							/* if( do_attenuation )
							{
								float len=L.norm( );
								mult=1.0/(c0+c1*len+c2*len*len);
							}
							else */
							{
								mult=1.0;
							}
							if( (int(floor(1024+intersection.get_x_comp( )))%2==0 && int(floor(1024+intersection.get_z_comp( )))%2==0) ||
								(int(floor(1024+intersection.get_x_comp( )))%2==1 && int(floor(1024+intersection.get_z_comp( )))%2==1) )
							{
								pix.r+=(1.0*r_scale*mult*light.get_intensity( ));
								pix.g+=(1.0*g_scale*mult*light.get_intensity( ));
								pix.b+=(1.0*b_scale*mult*light.get_intensity( ));
							}
							if( (int(floor(1024+intersection.get_x_comp( )))%2==0 && int(floor(1024+intersection.get_z_comp( )))%2==1) ||
								(int(floor(1024+intersection.get_x_comp( )))%2==1 && int(floor(1024+intersection.get_z_comp( )))%2==0) )
							{
								pix.r+=(0.0);
								pix.g+=(0.0);
								pix.b+=(0.0);
							}
						}
					}
					/* if( do_global )
					{
						if( do_indirect )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree, points,
								r_scale, g_scale, b_scale, num_photons, do_cone_filter, 0 );
						}
						if( do_spec )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_spec, points_spec,
								r_scale, g_scale, b_scale, spec_photons, do_cone_filter, 0 );
						}
						if( do_caust )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_caust, points_caust,
								r_scale, g_scale, b_scale, caust_photons, do_cone_filter, 1 );
						}
					} */
				}
				if( triangles[final_index].get_material( ).k_sr>0.0 || triangles[final_index].get_material( ).k_sg>0.0 ||
					triangles[final_index].get_material( ).k_sb>0.0 )
				{
					R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
						cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
					if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
					{
						N=(-1.0)*N;
					}
					N=N.normalized( );

					cop_direction=(intersection-e).normalized( )-2*((intersection-e).normalized( )).
						dot(N)*N;
					e=intersection+.01*cop_direction;
					recursion_depth++;
					illumination( pix, e, cop_direction, spheres, triangles, lloc, kd_tree_direct, points_direct, kd_tree_indirect, points_indirect, 
						kd_tree_caust, points_caust, triangles[final_index].get_material( ).k_sr*r_scale, triangles[final_index].get_material( ).k_sg*g_scale, 
						triangles[final_index].get_material( ).k_sb*b_scale, num_photons_d, num_photons_i, num_photons_c, do_global, direct, light, light_samples, 
						recursion_depth, do_cone_filter, do_attenuation, do_direct, do_indirect, do_caust, do_cone_filter_all );
					recursion_depth--;
				}
				if( triangles[final_index].get_material( ).k_tr>0.0 || triangles[final_index].get_material( ).k_tg>0.0 ||
					triangles[final_index].get_material( ).k_tb>0.0 )
				{
					R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
						cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
					N=N.normalized( );
					if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
					{
						N=(-1.0)*N;
					}
					R3_vector view=(intersection-e).normalized( );
					float mult=1.0/triangles[final_index].get_material( ).ref_index;
					float sin2_ang=pow(mult,2)*(1-pow(view.dot(N),2));
					cop_direction=mult*view+(mult*view.dot(N)-sqrt(1-sin2_ang))*N;
					cop_direction=cop_direction.normalized( );
					e=intersection+0.01*cop_direction;
					recursion_depth++;
					illumination( pix, e, cop_direction, spheres, triangles, lloc, kd_tree_direct, points_direct, kd_tree_indirect, points_indirect, 
						kd_tree_caust, points_caust, triangles[final_index].get_material( ).k_tr*r_scale, triangles[final_index].get_material( ).k_tg*g_scale, 
						triangles[final_index].get_material( ).k_tb*b_scale, num_photons_d, num_photons_i, num_photons_c, do_global, direct, light, light_samples, 
						recursion_depth, do_cone_filter, do_attenuation, do_direct, do_indirect, do_caust, do_cone_filter_all );
					recursion_depth--;
				}
				if( triangles[final_index].get_material( ).k_dr>0.0 || triangles[final_index].get_material( ).k_dg>0.0 ||
					triangles[final_index].get_material( ).k_db>0.0 )
				{
					if( direct )
					{
						if( light_samples > 1 )
						{
							int shadows=0;
							for( int i=0; i<4; i++ )
							{
								switch( i )
								{
									case 0: lloc=light.get_ll_corner( );
										break;
									case 1: lloc=light.get_lr_corner( );
										break;
									case 2: lloc=light.get_ul_corner( );
										break;
									case 3: lloc=light.get_ur_corner( );
										break;
								}
								R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
									cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
								t=-1;                

								N=N.normalized( );
								if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
								{
									N=(-1.0)*N;
								}
								if(  N.dot( lloc-intersection ) >= 0 )
								{
									R3_vector d=lloc-intersection;
									int i=0;
									while( i < t_size && (t >= 1 || t <= 0) )
									{
										if( i != final_index )
										{
											t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
										}
										i++;
									}
									i=0;
									while( i < s_size && (t >= 1 || t <= 0) )
									{
										t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
										i++;
									}
									if( t <= 1 && t >= 0 )
									{
										shadows++;
									}
								}
							}
							if( shadows > 0 && shadows < 4 )
							{
								float r=0, g=0, b=0;
								for( int i=0; i<light_samples; i++ )
								{
									lloc=light.get_location( );
									R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
										cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
									t=-1;                

									N=N.normalized( );
									if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
									{
										N=(-1.0)*N;
									}
									if(  N.dot( lloc-intersection ) >= 0 )
									{
										R3_vector d=lloc-intersection;
										int i=0;
										while( i < t_size && (t >= 1 || t <= 0) )
										{
											if( i != final_index )
											{
												t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
											}
											i++;
										}
										i=0;
										while( i < s_size && (t >= 1 || t <= 0) )
										{
											t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
											i++;
										}
										if( t >= 1 || t <= 0 )
										{
											R3_vector L=lloc-intersection;
											float mult;
											if( do_attenuation )
											{
												float len=L.norm( );
												mult=1.0/(c0+c1*len+c2*len*len);
											}
											else
											{
												mult=1.0;
											}
											L=L.normalized( );
											R3_vector V=e-intersection;
											V=V.normalized( );
											R3_vector H=L+V;
											H=H.normalized( );
											r += r_scale*mult*
												((triangles[final_index].get_material( ).k_dr)*(N.dot( L ))+
												(triangles[final_index].get_material( ).k_sr)*pow(H.dot( N ), 
												triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
											g += g_scale*mult*
												((triangles[final_index].get_material( ).k_dg)*(N.dot( L ))+
												(triangles[final_index].get_material( ).k_sg)*pow(H.dot( N ),
												triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
											b += b_scale*mult*
												((triangles[final_index].get_material( ).k_db)*(N.dot( L ))+
												(triangles[final_index].get_material( ).k_sb)*pow(H.dot( N ), 
												triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
										}
									}
								}
								pix.r+=(r/light_samples);
								pix.g+=(g/light_samples);
								pix.b+=(b/light_samples);
							}
							if( shadows==0 )
							{
								lloc=light.get_center( );
								R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
									cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
								t=-1;                

								N=N.normalized( );
								if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
								{
									N=(-1.0)*N;
								}
								if(  N.dot( lloc-intersection ) >= 0 )
								{
									R3_vector d=lloc-intersection;
									int i=0;
									while( i < t_size && (t >= 1 || t <= 0) )
									{
										if( i != final_index )
										{
											t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
										}
										i++;
									}
									i=0;
									while( i < s_size && (t >= 1 || t <= 0) )
									{
										t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
										i++;
									}
									if( t >= 1 || t <= 0 )
									{
										R3_vector L=lloc-intersection;
										float mult;
										if( do_attenuation )
										{
											float len=L.norm( );
											mult=1.0/(c0+c1*len+c2*len*len);
										}
										else
										{
											mult=1.0;
										}	
										L=L.normalized( );
										R3_vector V=e-intersection;
										V=V.normalized( );
										R3_vector H=L+V;
										H=H.normalized( );
										pix.r += r_scale*mult*
											((triangles[final_index].get_material( ).k_dr)*(N.dot( L ))+
											(triangles[final_index].get_material( ).k_sr)*pow(H.dot( N ), 
											triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
										pix.g += g_scale*mult*
											((triangles[final_index].get_material( ).k_dg)*(N.dot( L ))+
											(triangles[final_index].get_material( ).k_sg)*pow(H.dot( N ),
											triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
										pix.b += b_scale*mult*
											((triangles[final_index].get_material( ).k_db)*(N.dot( L ))+
											(triangles[final_index].get_material( ).k_sb)*pow(H.dot( N ), 
											triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
									}
								}
							}
						}
						else
						{
							R3_vector N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
								cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );
							t=-1;                

							N=N.normalized( );
							if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
							{
								N=(-1.0)*N;
							}
							if(  N.dot( lloc-intersection ) >= 0 )
							{
								R3_vector d=lloc-intersection;
								int i=0;
								while( i < t_size && (t >= 1 || t <= 0) )
								{
									if( i != final_index )
									{
										t=find_intersection_parameter_triangle( intersection, d, triangles[i] );
									}
									i++;
								}
								i=0;
								while( i < s_size && (t >= 1 || t <= 0) )
								{
									t=find_intersection_parameter_sphere( intersection, d, spheres[i] );
									i++;
								}
								if( t >= 1 || t <= 0 )
								{
									R3_vector L=lloc-intersection;
									float mult;
									if( do_attenuation )
									{
										float len=L.norm( );
										mult=1.0/(c0+c1*len+c2*len*len);
									}
									else
									{
										mult=1.0;
									}
									L=L.normalized( );
									R3_vector V=e-intersection;
									V=V.normalized( );
									R3_vector H=L+V;
									H=H.normalized( );
									pix.r += r_scale*mult*
										((triangles[final_index].get_material( ).k_dr)*(N.dot( L ))+
										(triangles[final_index].get_material( ).k_sr)*pow(H.dot( N ), 
										triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
									pix.g += g_scale*mult*
										((triangles[final_index].get_material( ).k_dg)*(N.dot( L ))+
										(triangles[final_index].get_material( ).k_sg)*pow(H.dot( N ),
										triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
									pix.b += b_scale*mult*
										((triangles[final_index].get_material( ).k_db)*(N.dot( L ))+
										(triangles[final_index].get_material( ).k_sb)*pow(H.dot( N ), 
										triangles[final_index].get_material( ).n_spec))*light.get_intensity( );
								}
								/* else
								{
								diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree, 
								points, r_scale, g_scale, b_scale, 64 );
								}
								}
								else
								{
								diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree, points,
								r_scale, g_scale, b_scale, 64 );
								} */
							}
						}
					}
					if( do_global )
					{
						if( do_direct )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_direct, points_direct,
								r_scale, g_scale, b_scale, num_photons_d, do_cone_filter_all, 0 );
						}
						if( do_indirect )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_indirect, points_indirect,
								r_scale, g_scale, b_scale, num_photons_i, do_cone_filter_all, 0 );
						}
						if( do_caust )
						{
							diffuse_illumination_alt( pix, type, final_index, spheres, triangles, intersection, e, kd_tree_caust, points_caust,
								r_scale, g_scale, b_scale, num_photons_c, do_cone_filter, 1 );
						}
					}
				}
			}
			index=0;
			type='a';
			final_index=0;
			t=-1;
		}
	}
}

void pmf::locate_photons( R3_vector& intersection, std::vector<Photon*>& kd_tree, 
						  std::priority_queue<hs::candidate_photon,std::vector<hs::candidate_photon>, 
						  Candidate_Photon_Comparator>& photons, float& R, int index )
{
	float distance=0.0;
	if( unsigned(2*index+2) < kd_tree.size( ) )
	{
		if( kd_tree[index]->get_plane( )==0 )
		{
			distance=intersection.get_x_comp( )-kd_tree[index]->get_position( ).get_x_comp( );
		}
		if( kd_tree[index]->get_plane( )==1 )
		{
			distance=intersection.get_y_comp( )-kd_tree[index]->get_position( ).get_y_comp( );
		}
		if( kd_tree[index]->get_plane( )==2 )
		{
			distance=intersection.get_z_comp( )-kd_tree[index]->get_position( ).get_z_comp( );
		}
		if( distance < 0 )
		{
			locate_photons( intersection, kd_tree, photons, R, 2*index+1 );
			if( distance*distance < R )
			{
				locate_photons( intersection, kd_tree, photons, R, 2*index+2 );
			}
		}
		else
		{
			locate_photons( intersection, kd_tree, photons, R, 2*index+2 );
			if( distance*distance < R )
			{
				locate_photons( intersection, kd_tree, photons, R, 2*index+1 );
			}
		}
	}

	distance=(intersection-kd_tree[index]->get_position( )).dot( intersection-kd_tree[index]->get_position( ) );
	if( distance < R )
	{
		hs::candidate_photon photon( kd_tree[index], distance );
		/// if( photons.size( )==29 )
		// {
		// 	std::cout << photons.size( ) << "\n";
		// }
		photons.push( photon );
		if( photons.size( ) > 256 )
		{
			// std::cout << "poop\n\n";
			R=photons.top( ).dist;
			photons.pop( );
		}
	}
}

void pmf::diffuse_illumination( hs::RGB& pix, char type, int final_index, std::vector<Sphere>& spheres, std::vector<Triangle>& triangles, 
								R3_vector& intersection, R3_vector& e, std::vector<Photon*>& kd_tree, float r_scale, float g_scale, 
								float b_scale  )
{
	
	float R=1.5;
	float R_squared=R*R;
	std::priority_queue<hs::candidate_photon,std::vector<hs::candidate_photon>, Candidate_Photon_Comparator> photons;
	locate_photons( intersection, kd_tree, photons, R_squared, 0 );

	/* if( photons.size( ) > 10 )
	{
		std::cout << "More than ten photons.\n\n";
	} */
		
	R3_vector N, dir;
	float multiplier=1/(3.14159*3.14159*R_squared);
	float r_power, g_power, b_power;
	if( photons.size( ) > 0 )
	{
		r_power=photons.top( ).photon->get_r_power( );
		g_power=photons.top( ).photon->get_g_power( );
		b_power=photons.top( ).photon->get_b_power( );
	}
	float r=0, g=0, b=0;
	while( !photons.empty( ) )
	{
		dir=photons.top( ).photon->get_direction( );
		photons.pop( );
		if( type == 't' )
		{
			N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
				cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );                
			N=N.normalized( );
			if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
			{
				N=(-1.0)*N;
			}	
			if( N.dot( dir ) < 0 )
			{
				dir=(-1.0)*dir;
			}
			if( triangles[final_index].get_material( ).type==2 )
			{
				float n=snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))*
							sin(8.0/exp(snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))));
				vec3 C=vec3(0.5*snoise(s*vec3(0.5*intersection.get_x_comp( ), 0.5*intersection.get_y_comp( ), 0.5*intersection.get_z_comp( )))+0.5,
						0.5*snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))+0.5,
						0.5*snoise(s*vec3(3.0/2.0*intersection.get_x_comp( ), 3.0/2.0*intersection.get_y_comp( ), 3.0/2.0*intersection.get_z_comp( )))+0.5);

				vec3 color=n*C*(N.dot(dir));

				r += r_power*color[0];
				g += g_power*color[1];
				b += b_power*color[2];
			}
			else
			{
				r+=(triangles[final_index].get_material( ).k_dr*(r_power*(N.dot(dir))));
				g+=(triangles[final_index].get_material( ).k_dg*(g_power*(N.dot(dir))));
				b+=(triangles[final_index].get_material( ).k_db*(b_power*(N.dot(dir))));
			}
		}
		else
		{
			N=intersection-spheres[final_index].get_center( );
			N=N.normalized( );
			if( N.dot( dir ) < 0 )
			{
				dir=(-1.0)*dir;
			}
			if( spheres[final_index].get_material( ).type==2 )
			{
				float s=.5;
				float n=snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))*
							sin(8.0/exp(snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))));
				vec3 C=vec3(0.5*snoise(vec3(s*0.5*intersection.get_x_comp( ), s*0.5*intersection.get_y_comp( ), s*0.5*intersection.get_z_comp( )))+0.5,
						0.5*snoise(vec3(s*intersection.get_x_comp( ), s*intersection.get_y_comp( ), s*intersection.get_z_comp( )))+0.5,
						0.5*snoise(s*vec3(3.0/2.0*intersection.get_x_comp( ), 3.0/2.0*intersection.get_y_comp( ), 3.0/2.0*intersection.get_z_comp( )))+0.5);

				vec3 color=n*C*r_power*(N.dot(dir));

				r += color[0];
				g += color[1];
				b += color[2];
			}
			else
			{
				r+=(spheres[final_index].get_material( ).k_dr*(r_power*(N.dot(dir))));
				g+=(spheres[final_index].get_material( ).k_dg*(g_power*(N.dot(dir))));
				b+=(spheres[final_index].get_material( ).k_db*(b_power*(N.dot(dir))));
			}
		}
	}
	pix.r+=r_scale*multiplier*r;
	pix.g+=g_scale*multiplier*g;
	pix.b+=b_scale*multiplier*b;
}

void pmf::diffuse_illumination_alt( hs::RGB& pix, char type, int final_index, std::vector<Sphere>& spheres, std::vector<Triangle>& triangles, 
								R3_vector& intersection, R3_vector& e, ANNkd_tree* kdTree, Point* points, float r_scale, float g_scale, 
								float b_scale, int photons_to_use, bool do_cone_filter, bool on_texture )
{
	
	// float R=1.5;
	// float R_squared=R*R;
	// std::priority_queue<hs::candidate_photon,std::vector<hs::candidate_photon>, Candidate_Photon_Comparator> photons;
	// locate_photons( intersection, kd_tree, photons, R_squared, 0 );

	/* if( photons.size( ) > 10 )
	{
		std::cout << "More than ten photons.\n\n";
	} */

	double query[3];
	query[0]=intersection.get_x_comp( );
	query[1]=intersection.get_y_comp( );
	query[2]=intersection.get_z_comp( );
	ANNpoint queryPt=query;
	ANNidxArray nnIdx = new ANNidx[photons_to_use];
	ANNdistArray dists = new ANNdist[photons_to_use];

	kdTree->annkSearch( queryPt,photons_to_use,nnIdx,dists);
		
	R3_vector N, dir;
	float multiplier;
	float k=0;
	if( do_cone_filter )
	{
		k=1.1;
		// multiplier=1/((1-2.0/(3*k))*3.14159*dists[photons_to_use-1]);
		multiplier=1/((1-2.0/(3*k))*dists[photons_to_use-1]);
	}
	else
	{
		multiplier=1/(dists[photons_to_use-1]);
	}
	// std::cout << multiplier << "\n";
	float r_power, g_power, b_power;
	float r=0, g=0, b=0;
	for( int i=0; i<photons_to_use; i++ )
	{
		r_power=points[nnIdx[i]].photon->get_r_power( );
		g_power=points[nnIdx[i]].photon->get_g_power( );
		b_power=points[nnIdx[i]].photon->get_b_power( );
		float mult2;
		if( do_cone_filter )
		{
			mult2=1-dists[i]/(k*dists[photons_to_use-1]);
		}
		else
		{
			mult2=1.0;
		}
		dir=points[nnIdx[i]].photon->get_direction( );
		if( type == 't' )
		{
			N=(triangles[final_index].get_vertex2( )-triangles[final_index].get_vertex1( )).
				cross( triangles[final_index].get_vertex3( )-triangles[final_index].get_vertex1( ) );                
			N=N.normalized( );
			if( N.dot( e - triangles[final_index].get_vertex1( ) ) < 0 )
			{
				N=(-1.0)*N;
			}	
			if( N.dot( dir ) < 0 )
			{
				dir=(-1.0)*dir;
			}
			if( triangles[final_index].get_material( ).type==2 )
			{
				float n=snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))*
							sin(8.0/exp(snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))));
				vec3 C=vec3(0.5*snoise(s*vec3(0.5*intersection.get_x_comp( ), 0.5*intersection.get_y_comp( ), 0.5*intersection.get_z_comp( )))+0.5,
						0.5*snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))+0.5,
						0.5*snoise(s*vec3(3.0/2.0*intersection.get_x_comp( ), 3.0/2.0*intersection.get_y_comp( ), 3.0/2.0*intersection.get_z_comp( )))+0.5);

				vec3 color=n*C*(N.dot(dir));

				r += r_power*color[0];
				g += g_power*color[1];
				b += b_power*color[2];
			}
			else
			{
				if( triangles[final_index].get_material( ).type==3 )
				{
					if( (int(floor(1024+intersection.get_x_comp( )))%2==0 && int(floor(1024+intersection.get_z_comp( )))%2==0) ||
						(int(floor(1024+intersection.get_x_comp( )))%2==1 && int(floor(1024+intersection.get_z_comp( )))%2==1) )
					{
						r+=mult2*(r_power*(N.dot(dir)));
						g+=mult2*(g_power*(N.dot(dir)));
						b+=mult2*(b_power*(N.dot(dir)));
					}
					if( (int(floor(1024+intersection.get_x_comp( )))%2==0 && int(floor(1024+intersection.get_z_comp( )))%2==1) ||
						(int(floor(1024+intersection.get_x_comp( )))%2==1 && int(floor(1024+intersection.get_z_comp( )))%2==0) )
					{
						if( on_texture )
						{
							r+=mult2*(r_power*(N.dot(dir)));
							g+=mult2*(g_power*(N.dot(dir)));
							b+=mult2*(b_power*(N.dot(dir)));
						}
						else
						{
							r+=0.0;
							g+=0.0;
							b+=0.0;
						}
					}
				}
				else
				{
					r+=mult2*(triangles[final_index].get_material( ).k_dr*(r_power*(N.dot(dir))));
					g+=mult2*(triangles[final_index].get_material( ).k_dg*(g_power*(N.dot(dir))));
					b+=mult2*(triangles[final_index].get_material( ).k_db*(b_power*(N.dot(dir))));
					/* r+=mult2*(r_power*(N.dot(dir)));
					g+=mult2*(g_power*(N.dot(dir)));
					b+=mult2*(b_power*(N.dot(dir))); */
				}
			}
		}
		else
		{
			N=intersection-spheres[final_index].get_center( );
			N=N.normalized( );
			if( N.dot( dir ) < 0 )
			{
				dir=(-1.0)*dir;
			}
			if( spheres[final_index].get_material( ).type==2 )
			{
				float s=.5;
				float n=snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))*
							sin(8.0/exp(snoise(s*vec3(intersection.get_x_comp( ), intersection.get_y_comp( ), intersection.get_z_comp( )))));
				vec3 C=vec3(0.5*snoise(vec3(s*0.5*intersection.get_x_comp( ), s*0.5*intersection.get_y_comp( ), s*0.5*intersection.get_z_comp( )))+0.5,
						0.5*snoise(vec3(s*intersection.get_x_comp( ), s*intersection.get_y_comp( ), s*intersection.get_z_comp( )))+0.5,
						0.5*snoise(s*vec3(3.0/2.0*intersection.get_x_comp( ), 3.0/2.0*intersection.get_y_comp( ), 3.0/2.0*intersection.get_z_comp( )))+0.5);

				vec3 color=n*C*r_power*(N.dot(dir));

				r += color[0];
				g += color[1];
				b += color[2];
			}
			else
			{
				r+=mult2*(spheres[final_index].get_material( ).k_dr*(r_power*(N.dot(dir))));
				g+=mult2*(spheres[final_index].get_material( ).k_dg*(g_power*(N.dot(dir))));
				b+=mult2*(spheres[final_index].get_material( ).k_db*(b_power*(N.dot(dir))));
				/* r+=mult2*(r_power*(N.dot(dir)));
				g+=mult2*(g_power*(N.dot(dir)));
				b+=mult2*(b_power*(N.dot(dir))); */
			}
		}
	}
	pix.r+=(r_scale*multiplier*r);
	pix.g+=(g_scale*multiplier*g);
	pix.b+=(b_scale*multiplier*b);

	delete [] nnIdx;
	delete [] dists;
}