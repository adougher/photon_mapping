#include "Image.h"
#include "helper_structs.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <assert.h>

/* ----------- image class: methods ---------- */

Image::Image( int m, int n ):xsize( m ), ysize( n )
{
  	rgb = new hs::RGB[m*n];
}

/* ----------------------- */

hs::RGB& Image::pixel ( int i, int j )
{
  	return rgb[ i+xsize*j ];
}

/* ----------------------- */

unsigned char Image::clampnround ( double x )
{
	if( x>255 )
	{
    		x = 255;
	}
  	if( x<0 )
	{ 
    		x = 0;
	}
  	return (unsigned char)floor(x+.5);
}

/* ----------------------- */

void Image::save_to_ppm_file ( char *filename )
{
  	std::ofstream ofs(filename,std::ios::binary);
  	assert(ofs);
  	ofs << "P6" << std::endl;
  	ofs << xsize << " " << ysize << std::endl << 255 << std::endl;
  	for ( int i=0; i<xsize*ysize; i++ )
    	{
      		unsigned char r = clampnround(256*rgb[i].r);
      		unsigned char g = clampnround(256*rgb[i].g);
      		unsigned char b = clampnround(256*rgb[i].b);
      		ofs.write((char*)&r,sizeof(char));
      		ofs.write((char*)&g,sizeof(char));
      		ofs.write((char*)&b,sizeof(char));
    	}
}
