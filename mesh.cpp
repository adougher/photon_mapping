
#include "mesh.h"
#include <fstream>
#include <glm/glm.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace glm;

namespace EZGraphics {

/* ------------------------------------------- */

Mesh::Mesh ( float* CoordArray, int number_of_vertices )
{
	vertices=number_of_vertices;
	vtable = new vec3 [vertices];

	for ( int i=0; i<vertices; i++ )
	{
		vtable[i][0]=CoordArray[3*i]; 
		vtable[i][1]=CoordArray[3*i+1];
		vtable[i][2]=CoordArray[3*i+2];;
		if (!i)
		{
			bbox[0] = bbox[1] = vtable[0];
		}
		else
		{
			for ( int j=0; j<3; j++ )
			{
				if (vtable[i][j]<bbox[0][j])
				{
					bbox[0][j] = vtable[i][j];
				}
				if (vtable[i][j]>bbox[1][j])
				{
					bbox[1][j] = vtable[i][j];
				}
			}
	    }
    }
    
	center = 0.5f*(bbox[0]+bbox[1]);
	vec3 boxsize = bbox[1]-bbox[0];
	bboxmaxdim = glm::max(glm::max(boxsize[0],boxsize[1]),boxsize[2]);
}

/* ------------------------------------------- */

Mesh::~Mesh()
{
  if (vtable) delete[] vtable;
}

/* ------------------------------------------- */


glm::vec3 *Mesh::getVertexTable()
{
  return vtable;
}

/* ------------------------------------------- */

/* ------------------------------------------- */

vec3 Mesh::getUpperCorner()
{
  return bbox[1];
}

/* ------------------------------------------- */

vec3 Mesh::getLowerCorner()
{
  return bbox[0];
}

/* ------------------------------------------- */

vec3 Mesh::getCenter()
{
  return center;
}

/* ------------------------------------------- */

float Mesh::getDiameter()
{
  return bboxdiam;
}

/* ------------------------------------------- */

float Mesh::getMaxDim()
{
  return bboxmaxdim;
}

/* ------------------------------------------- */

int Mesh::getVertexCount()
{
  return vertices;
}

/* ------------------------------------------- */

  static bool _cmp_ivec4 ( const ivec4 &a, const ivec4 &b )
  {
    if (a[0]<b[0]) return true;
    if (a[0]>b[0]) return false;
    if (a[1]<b[1]) return true;
    if (a[1]>b[1]) return false;
    if (a[2]<b[2]) return true;
    if (a[2]>b[2]) return false;
    if (a[3]<b[3]) return true;
    return false;
  }

/* ------------------------------------------- */
};
