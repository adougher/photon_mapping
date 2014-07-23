#include <vector>
#include <queue>
#include "Sphere.h"
#include "Triangle.h"
#include "Photon.h"
#include "Light_source.h"
#include "helper_structs.h"
#include "point.h"

#include <ANN.h>

#ifndef PHOTON_MAPPING_FUNCTIONS_H
#define PHOTON_MAPPING_FUNCTIONS_H

namespace pmf
{
	struct Candidate_Photon_Comparator 
	{
		bool operator()( const hs::candidate_photon c1, const hs::candidate_photon c2 ) const
		{
			return c1.dist < c2.dist;
		}
	};

	void read_input_file( int& resolution_x, int& resolution_y, float* viewpoint, float* screen_lower_left_corner, float* screen_horizontal_vector, 
						  float* screen_vertical_vector, std::vector<Light_source>& lights, float& ambient_light_intensity, 
						  std::vector<Sphere>& spheres, std::vector<Triangle>& triangles );
	float find_intersection_parameter_sphere( R3_vector& origin, R3_vector& direction, Sphere& sphere );
	float find_intersection_parameter_triangle( R3_vector& origin, R3_vector& direction, Triangle& triangle );
	char get_power_expon( float num );
	void emit_photons( std::vector<Light_source>& lights, unsigned long int total_photons, bool do_direct, std::vector<Photon*>& direct_photons, 
						bool do_indirect, std::vector<Photon*>& indirect_photons, bool do_caust, std::vector<Photon*>& caust_photons, 
						std::vector<Sphere>& spheres, std::vector<Triangle>& triangles, bool do_ray_trace, bool one_map, bool do_color_bleeding );
	int partition( std::vector<Photon*>& photons, Photon* pho, int p, int r, short& axis );
	Photon* get_median( std::vector<Photon*>& photons, int p, int r, int i, short& axis );
	Photon* get_median_large( std::vector<Photon*>& photons, int p, int r, int i, short& axis );
	void build_photon_map( std::vector<Photon*>& kd_tree, std::vector<Photon*>& photons, int n );
	void build_photon_map_large( std::vector<Photon*>& kd_tree, std::vector<Photon*>& photons, int n, int p );
	void illumination( hs::RGB& pix, R3_vector e, R3_vector cop_direction, std::vector<Sphere>& spheres, std::vector<Triangle>& triangles, 
						R3_vector lloc, ANNkd_tree* kd_tree_direct, Point* points_direct, ANNkd_tree* kd_tree_indirect, Point* points_indirect,
						ANNkd_tree* kd_tree_caust, Point* points_caust, float r_scale, float g_scale, float b_scale, int num_photons_d, 
						int num_photons_i, int num_photons_c, bool do_global, bool direct, Light_source light, int light_samples,
						int& recursion_depth, bool do_cone_filter, bool do_attenuation, bool do_direct, bool do_indirect, 
						bool do_caust, bool do_cone_filter_all );

					   // std::vector<Photon*>& kd_tree, float r_scale, float g_scale, float b_scale );
	void locate_photons( R3_vector& intersection, std::vector<Photon*>& kd_tree, 
						 std::priority_queue<hs::candidate_photon,std::vector<hs::candidate_photon>, 
						 Candidate_Photon_Comparator>& photons, float& R, int index );
	void diffuse_illumination( hs::RGB& pixel, char type, int final_index, std::vector<Sphere>& spheres, 
							   std::vector<Triangle>& triangles, R3_vector& intersection, 
							   R3_vector& e, std::vector<Photon*>& kd_tree, float r_scale, float g_scale, 
							   float b_scale );
	void diffuse_illumination_alt( hs::RGB& pixel, char type, int final_index, std::vector<Sphere>& spheres, 
								   std::vector<Triangle>& triangles, R3_vector& intersection, 
							       R3_vector& e, ANNkd_tree* kdTree, Point* points, float r_scale, float g_scale, 
							       float b_scale, int photons_to_use, bool do_cone_filter, bool on_texture );
};
#endif