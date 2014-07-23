#include "Photon.h"

#ifndef HELPER_STRUCTS_H
#define HELPER_STRUCTS_H

namespace hs
{
	struct RGB 
	{
  		float r,g,b;
	};

	struct material
	{
		material( ):k_dr( 1 ), k_dg( 1 ), k_db( 1 ), k_sr( 1 ), k_sg( 1 ), k_sb( 1 ), k_tr( 1 ), k_tg( 1 ), k_tb( 1 ), ref_index( 1 ), n_spec( 1 ), type( 0 ) { };
		material( float kdr, float kdg, float kdb, float ksr, float ksg, float ksb, float ktr, float ktg, float ktb, float ri, float nspec, short type ):
						k_dr( kdr ), k_dg( kdg ), k_db( kdb ), k_sr( ksr ), k_sg( ksg ), k_sb( ksb ), k_tr( ktr ), k_tg( ktg ), k_tb( ktb ), ref_index( ri ), n_spec( nspec ), type( type ) { };
		float k_dr, k_dg, k_db, k_sr, k_sg, k_sb, k_tr, k_tg, k_tb, ref_index, n_spec;
		short type;
	};	
	
	struct candidate_photon
	{
		candidate_photon( Photon* p, float d ): photon( p ), dist( d ) { };

		Photon* photon;
		float dist;
	};
}
#endif
