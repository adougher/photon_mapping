#include "Sphere.h"
#include <iostream>
#include <cmath>

std::ostream& operator<<( std::ostream& output, const Sphere& s )
{
        output << "(" << s.center << ", " << s.radius << ")";
        return output;
}
