
#include <glm/glm.hpp>
#include <memory>

#pragma once

namespace EZGraphics {

  /* ------------------------------------------- */

  class Mesh {

    int vertices, triangles;
    glm::vec3* vtable;
    glm::vec3 bbox[2];
    glm::vec3 center;
    float bboxdiam;
    float bboxmaxdim;

  public:
  
    Mesh ( float* CoordArray, int num_vertices );
    ~Mesh();

    glm::vec3 *getVertexTable();

    glm::vec3 getUpperCorner();
    glm::vec3 getLowerCorner();
    glm::vec3 getCenter();
    float getDiameter();
    float getMaxDim();
    int getVertexCount();
    
  };
  
  /* ------------------------------------------- */
  
};
