#include "helper_structs.h"

#ifndef PRIMITIVE_H
#define PRIMITIVE_H

class Primitive
{
public:
	Primitive( ):material_data( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ) { };
	Primitive( hs::material mat ): material_data( mat ) { };
	void set_material( hs::material mat ) { material_data=mat; };
	hs::material get_material( ) { return material_data; };
protected:
	hs::material material_data;
};
#endif
