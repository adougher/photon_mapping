#include "helper_structs.h"

#ifndef IMAGE_H
#define IMAGE_H

class Image 
{
public:
    Image ( int m, int n );       		// allocates image of specified size
    hs::RGB &pixel ( int i, int j );  	// access to a specific pixel
	unsigned char clampnround ( double x );
    void save_to_ppm_file ( char *filename );
private:
	int xsize,ysize; 					// resolution
    hs::RGB *rgb;        				// pixel intensities
};
#endif
