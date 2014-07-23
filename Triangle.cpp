#include "Triangle.h"
#include <iostream>
#include <cmath>

std::ostream& operator<<( std::ostream& output, const Triangle& t )
{
        output << "(" << t.vertex1 << ", " << t.vertex2 << ", " << t.vertex3 << ")";
        return output;
}
