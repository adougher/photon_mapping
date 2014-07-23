#include <cstdlib>
#include <math.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <vector>
#include "helper_structs.h"
#include "Primitive.h"
#include "R3_vector.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Photon.h"
#include "Light_source.h"
#include "Image.h"
#include "photon_mapping_functions.h"
#include "point.h"

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
#include "framebuffer.h"

using namespace std;
using namespace glm;
using namespace EZGraphics;

float* CoordArray_d;
float* CoordArray_i;
float* CoordArray_c;
int number_of_vertices_d, number_of_vertices_i, number_of_vertices_c;

class ViewerEventHandlers : public TrackballHandler, public MenuCreator {
	VertexArray *va_d, *va_i, *va_c;
	Buffer *loc_d, *loc_i, *loc_c;

	float maxdim_d, r_d, maxdim_i, r_i, maxdim_c, r_c;
	vec3 center_d, center_i, center_c;

	Program *pgmPoint;

	public:

	ViewerEventHandlers ( int argc, char **argv ) :
    TrackballHandler(argc,argv,GLUT_DEPTH|GLUT_DOUBLE|GLUT_RGB,800,800)
	{
	}
	 
	virtual void initializeGL( )
	{
		cout << "Creating program..." << endl;
		pgmPoint = createProgram(
				    ShaderFile(Vert,"C:/Users/Andrew/Documents/School/Computer_Science/Advanced_Graphics/Final_project/shader/vtx.glsl"),
					ShaderFile(Frag,"C:/Users/Andrew/Documents/School/Computer_Science/Advanced_Graphics/Final_project/shader/frg.glsl")
				    );
		Mesh Md(CoordArray_d, number_of_vertices_d);
		center_d = Md.getCenter();
		maxdim_d = Md.getMaxDim();
		r_d = length(Md.getUpperCorner()-Md.getLowerCorner())/(2*maxdim_d);
		loc_d = new Buffer(Md.getVertexCount(),Md.getVertexTable());
		va_d = new VertexArray;
		va_d->attachAttribute(0,loc_d);

		Mesh Mi(CoordArray_i, number_of_vertices_i);
		center_i = Mi.getCenter();
		maxdim_i = Mi.getMaxDim();
		r_i = length(Mi.getUpperCorner()-Mi.getLowerCorner())/(2*maxdim_i);
		loc_i = new Buffer(Mi.getVertexCount(),Mi.getVertexTable());
		va_i = new VertexArray;
		va_i->attachAttribute(0,loc_i);

		Mesh Mc(CoordArray_c, number_of_vertices_c);
		center_c = Mc.getCenter();
		maxdim_c = Mc.getMaxDim();
		r_c = length(Mc.getUpperCorner()-Mc.getLowerCorner())/(2*maxdim_c);
		loc_c = new Buffer(Mc.getVertexCount(),Mc.getVertexTable());
		va_c = new VertexArray;
		va_c->attachAttribute(0,loc_c);
	}

	virtual void draw()
	{
		glClear(GL_COLOR_BUFFER_BIT);

		pgmPoint->setUniform("MV",
		    translate(mat4(),vec3(0.0f,0.0f,-20.0f)) *
		    mat4(getRotation()) *
		    scale(mat4(),vec3(1/maxdim_d)) *
		    translate(mat4(),-center_d)
		  );

		// place camera at the origin, facing -z; near/far clip planes 18 and 22 away.

		pgmPoint->setUniform("P",perspective(getZoom(),getAspectRatio(),18.0f,22.0f));
		pgmPoint->setUniform("rgb",vec3(0.0,1.0,0.0));

		glPointSize(1.0f);

		pgmPoint->on();

		va_d->sendToPipeline(GL_POINTS,0,number_of_vertices_d);

		pgmPoint->setUniform("MV",
		    translate(mat4(),vec3(0.0f,0.0f,-20.0f)) *
		    mat4(getRotation()) *
		    scale(mat4(),vec3(1/maxdim_i)) *
		    translate(mat4(),-center_i)
		  );

		// place camera at the origin, facing -z; near/far clip planes 18 and 22 away.

		pgmPoint->setUniform("rgb",vec3(0.0,0.0,1.0));

		va_i->sendToPipeline(GL_POINTS,0,number_of_vertices_i);

		pgmPoint->setUniform("MV",
		    translate(mat4(),vec3(0.0f,0.0f,-20.0f)) *
		    mat4(getRotation()) *
		    scale(mat4(),vec3(1/maxdim_c)) *
		    translate(mat4(),-center_c)
		  );

		pgmPoint->setUniform("rgb",vec3(1.0,0.0,0.0));

		va_c->sendToPipeline(GL_POINTS,0,number_of_vertices_c);

		pgmPoint->off();
	}
};


/* ----------- main function ---------- */

int main ( int argc, char *argv[] )
{
  	int x,y;

	int resolution_x, resolution_y;

	float viewpoint[3];
    float screen_lower_left_corner[3];
    float screen_horizontal_vector[3];
    float screen_vertical_vector[3];
    float ambient_light_intensity;

	vector<Light_source> lights;
	vector<Sphere> spheres;
	vector<Triangle> triangles;

  	pmf::read_input_file( resolution_x, resolution_y, viewpoint, screen_lower_left_corner, screen_horizontal_vector, 
							screen_vertical_vector, lights, ambient_light_intensity, spheres, triangles );

	R3_vector view_point( viewpoint[0], viewpoint[1], viewpoint[2] );
	R3_vector scr_low_left_corner( screen_lower_left_corner[0], screen_lower_left_corner[1], screen_lower_left_corner[2] );
	R3_vector scr_horiz_vec( screen_horizontal_vector[0], screen_horizontal_vector[1], screen_horizontal_vector[2] );
	R3_vector scr_vert_vec( screen_vertical_vector[0], screen_vertical_vector[1], screen_vertical_vector[2] );

	bool show_direct_photons=0;
	bool show_indirect_photons=0;
	bool show_caust_photons=0;
	bool do_ray_trace=1;
	int light_samples=1;
	int pixel_samples=1;
	bool do_global=1;
	bool do_cone_filter=1;
	bool do_cone_filter_all=1;
	bool do_attenuation=1;
	bool do_color_bleeding=0;

	bool do_direct=0;
	bool do_indirect=1;
	bool do_caust=1;
	bool do_one_map=0;

	unsigned long int total_photons=int(pow(2.0,15))-1;
	int radiance_est=512;
	int indirect_radiance_est=32;
	int caust_radiance_est=32;

	if( !do_global )
	{
		do_direct=0;
		do_indirect=0;
		do_caust=0;
		total_photons=0;
	}

	if( do_one_map )
	{
		do_indirect=0;
		do_caust=0;
	}
	else
	{
		if( !do_direct ) { do_direct=0; }
		if( !do_indirect ) { do_indirect=0; }
		if( !do_caust ) { do_caust=0; }
	}
	vector<Photon*> direct_photons;
	vector<Photon*> indirect_photons;
	vector<Photon*> caust_photons;

	// vector<Photon*> kd_tree(num_photons);

	pmf::emit_photons( lights, total_photons, do_direct, direct_photons, do_indirect, indirect_photons, do_caust, caust_photons,
						spheres, triangles, do_ray_trace, do_one_map, do_color_bleeding );

	unsigned long int num_direct_photons=direct_photons.size( );
	unsigned long int num_indirect_photons=indirect_photons.size( );
	unsigned long int num_caust_photons=caust_photons.size( );

	std::cout << "Direct Photons: " << num_direct_photons << endl;
	std::cout << "Indirect Photons: " << num_indirect_photons << endl;
	std::cout << "Caustic Photons: " << num_caust_photons << endl;

	// if( num_photons < 250000 )
	// {
	//	pmf::build_photon_map( kd_tree, photons, 0 );
	// }
	// else
	{
	//	pmf::build_photon_map_large( kd_tree, photons, 0, num_photons-1 );
	}

	if( show_direct_photons )
	{
		CoordArray_d = new float[3*num_direct_photons];
		number_of_vertices_d=num_direct_photons;
		for( unsigned int i=0; i<num_direct_photons; i++ )
		{
			CoordArray_d[3*i]=float(1.0*direct_photons[ i ]->get_position( ).get_x_comp( ));
			CoordArray_d[3*i+1]=float(1.0*direct_photons[ i ]->get_position( ).get_y_comp( ));
			CoordArray_d[3*i+2]=float(-1.0*direct_photons[ i ]->get_position( ).get_z_comp( ));
		}
	}
	if( show_indirect_photons )
	{
		CoordArray_i = new float[3*num_indirect_photons];
		number_of_vertices_i=num_indirect_photons;
		for( unsigned int i=0; i<num_indirect_photons; i++ )
		{
			CoordArray_i[3*i]=float(1.0*indirect_photons[ i ]->get_position( ).get_x_comp( ));
			CoordArray_i[3*i+1]=float(1.0*indirect_photons[ i ]->get_position( ).get_y_comp( ));
			CoordArray_i[3*i+2]=float(-1.0*indirect_photons[ i ]->get_position( ).get_z_comp( ));
		}
	}

	if( show_caust_photons )
	{
		CoordArray_c = new float[3*num_caust_photons];
		number_of_vertices_c=num_caust_photons;
		for( unsigned int i=0; i<num_caust_photons; i++ )
		{
			CoordArray_c[3*i]=float(1.0*caust_photons[ i ]->get_position( ).get_x_comp( ));
			CoordArray_c[3*i+1]=float(1.0*caust_photons[ i ]->get_position( ).get_y_comp( ));
			CoordArray_c[3*i+2]=float(-1.0*caust_photons[ i ]->get_position( ).get_z_comp( ));
		}
	}
	if( show_direct_photons || show_indirect_photons || show_caust_photons )
	{
		GLUTwrapper(new ViewerEventHandlers(argc,argv)).run();
	}

	ANNpointArray dataPts_direct=annAllocPts( num_direct_photons, 3 );
	Point* points_direct=new Point[3*num_direct_photons];
	for( unsigned int i=0; i<num_direct_photons; i++ )
	{
		points_direct[i].coords[3*i]=double(direct_photons[ i ]->get_position( ).get_x_comp( ));
		points_direct[i].coords[3*i+1]=double(direct_photons[ i ]->get_position( ).get_y_comp( ));
		points_direct[i].coords[3*i+2]=double(direct_photons[ i ]->get_position( ).get_z_comp( ));
		points_direct[i].photon=direct_photons[i];
		dataPts_direct[i]=points_direct[i].coords;
	}
	ANNkd_tree* kdTree_direct = new ANNkd_tree( dataPts_direct,num_direct_photons,3 ); 

	ANNpointArray dataPts_indirect=annAllocPts( num_indirect_photons, 3 );
	Point* points_indirect=new Point[3*num_indirect_photons];
	for( unsigned int i=0; i<num_indirect_photons; i++ )
	{
		points_indirect[i].coords[3*i]=double(indirect_photons[ i ]->get_position( ).get_x_comp( ));
		points_indirect[i].coords[3*i+1]=double(indirect_photons[ i ]->get_position( ).get_y_comp( ));
		points_indirect[i].coords[3*i+2]=double(indirect_photons[ i ]->get_position( ).get_z_comp( ));
		points_indirect[i].photon=indirect_photons[i];
		dataPts_indirect[i]=points_indirect[i].coords;
	}
	ANNkd_tree* kdTree_indirect = new ANNkd_tree( dataPts_indirect,num_indirect_photons,3 );

	ANNpointArray dataPts_caust=annAllocPts( num_caust_photons, 3 );
	Point* points_caust=new Point[3*num_caust_photons];
	for( unsigned int i=0; i<num_caust_photons; i++ )
	{
		points_caust[i].coords[3*i]=double(caust_photons[ i ]->get_position( ).get_x_comp( ));
		points_caust[i].coords[3*i+1]=double(caust_photons[ i ]->get_position( ).get_y_comp( ));
		points_caust[i].coords[3*i+2]=double(caust_photons[ i ]->get_position( ).get_z_comp( ));
		points_caust[i].photon=caust_photons[i];
		dataPts_caust[i]=points_caust[i].coords;
	}
	ANNkd_tree* kdTree_caust = new ANNkd_tree( dataPts_caust,num_caust_photons,3 );

	// ANNkd_tree::ANNkd_tree(
	// ANNpointArray pa, // data point array
	// int n, // number of points
    // int d); // dimension

	int l_size=lights.size( );
	int recursion_depth=0;
	R3_vector lloc, cop_direction;
  	Image img( resolution_x, resolution_y );
	if( light_samples==1 )
	{
		lloc=lights[0].get_center( );
	}
  	for ( x=0; x<resolution_x; x++ )
	{
    	for ( y=0; y<resolution_y; y++ )
      	{
			hs::RGB& pix = img.pixel(x,y);
			pix.r=0;
			pix.g=0;
			pix.b=0;
			if( pixel_samples==1 )
			{
				cop_direction=scr_low_left_corner+float((x+0.5)/float(resolution_x))*scr_horiz_vec+
							float((resolution_y-y-1+0.5)/float(resolution_y))*scr_vert_vec-view_point;
			}
			for( int i=0; i<l_size; i++ )
			{
				
				for( int j=0; j<pixel_samples; j++ )
				{
					if( pixel_samples > 1 )
					{
						cop_direction=scr_low_left_corner+float((x+1.0*rand( )/RAND_MAX)/float(resolution_x))*scr_horiz_vec+
							float((resolution_y-y-1+1.0*rand( )/RAND_MAX)/float(resolution_y))*scr_vert_vec-view_point;
					}
					pmf::illumination( pix, view_point, cop_direction, spheres, triangles, lloc, kdTree_direct, points_direct, 
						kdTree_indirect, points_indirect, kdTree_caust, points_caust, lights[i].get_intensity( ), lights[i].get_intensity( ), 
						lights[i].get_intensity( ), radiance_est, indirect_radiance_est, caust_radiance_est, do_global, do_ray_trace, 
						lights[i], light_samples, recursion_depth, do_cone_filter, do_attenuation, do_direct,
						do_indirect, do_caust, do_cone_filter_all );
				}
				pix.r/=(1.0*pixel_samples);
				pix.g/=(1.0*pixel_samples);
				pix.b/=(1.0*pixel_samples);
			}
							   
							   // kd_tree, 1.0, 1.0, 1.0 );	
      	}
		if( (x+1)%50==0 )
		{
			std::cout << "Column: " << x+1 << " completed.\n\n";
		}
	}

  	// img.save_to_ppm_file((char*)"C:/Users/Andrew/Documents/School/Computer_Science/Advanced_Graphics/Final_Project/p10l100g1g2_18_64s2_15_16c2_15_8.ppm");
	img.save_to_ppm_file((char*)"C:/Users/Andrew/Documents/School/Computer_Science/Advanced_Graphics/Final_Project/test.ppm");
  	return 0;
}