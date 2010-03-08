#include <math.h>
#include "llgalign.h"
#include "spinsystem.h"
#include "spinoperation.h"

LLGAlign::LLGAlign()
	: LLG("Align", ENCODE_LLGALIGN)
{
	gamma = 1.0;
}

bool LLGAlign::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto)
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

	for(int i=0; i<spinfrom->nxyz; i++)
	{
		const double h = sqrt(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);
		if(ms[i] > 0 && h > 0)
		{
			x[i] = ms[i] * hx[i] / h;
			y[i] = ms[i] * hy[i] / h;
			z[i] = ms[i] * hz[i] / h;
		}
	}

	spinto->time = spinfrom->time + dt;
	return true;
}









