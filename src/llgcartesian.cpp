#include <math.h>
#include "llgcartesian.h"
#include "spinsystem.h"
#include "spinoperation.h"
// Gilbert equation of motion:
//  dS    -g           a  g
//  -- = ----  S X h - ---- S X (S X h)
//  dt   1+aa        (1+aa)|S|
//
// or
//
// dS    -g           a
// -- = ---- S X (h +---S X H)
// dt   1+aa         |S|

LLGCartesian::LLGCartesian()
	: LLG("Cartesian", ENCODE_LLGCART)
{
	gamma = 1.0;
}




#define CROSS(v, a, b) \
	v[0] = a[1] * b[2] - a[2] * b[1]; \
	v[1] = a[2] * b[0] - a[0] * b[2]; \
	v[2] = a[0] * b[1] - a[1] * b[0];

bool LLGCartesian::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto)
{
	const double* sx = spinfrom->x;
	const double* sy = spinfrom->y;
	const double* sz = spinfrom->z;
	const double* ms = spinfrom->ms;

	const double* hx = fieldfrom->hx[SUM_SLOT];
	const double* hy = fieldfrom->hy[SUM_SLOT];
	const double* hz = fieldfrom->hz[SUM_SLOT];

	      double* x  = spinto->x;
	      double* y  = spinto->y;
	      double* z  = spinto->z;


	//LLG from http://inoe.inoe.ro/joam/arhiva/pdf8_5/5Ciubotaru.pdf
// 	#pragma omp parallel for
	for(int i=0; i<spinfrom->nxyz; i++)
	{
		if(ms[i] > 0)
		{
			double dM[3];
			double M[3];
			double H[3];
			double MH[3];
			double MMH[3];
			double gaa, inv;
	
			M[0]=sx[i];     M[1]=sy[i];     M[2]=sz[i];
			H[0]=hx[i];     H[1]=hy[i];     H[2]=hz[i];

			CROSS(MH, M, H);
			CROSS(MMH, M, MH);

			gaa = gamma / (1.0+alpha*alpha);

			for(int c=0; c<3; c++)
				dM[c] = -gaa * MH[c] - (alpha/ms[i]) * gaa * MMH[c];

			M[0] = (sx[i] + dt * dM[0]);
			M[1] = (sy[i] + dt * dM[1]);
			M[2] = (sz[i] + dt * dM[2]);

			inv = 1.0 / sqrt(M[0]*M[0] + M[1]*M[1] + M[2]*M[2]);

			x[i] = M[0] * inv * ms[i];
			y[i] = M[1] * inv * ms[i];
			z[i] = M[2] * inv * ms[i];
		}
	}

	spinto->time = spinfrom->time + dt;
}









