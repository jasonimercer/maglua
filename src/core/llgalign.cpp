/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include <math.h>
#include "llgalign.h"
#include "spinsystem.h"
#include "spinoperation.h"

LLGAlign::LLGAlign()
	: LLG("Align", ENCODE_LLGALIGN)
{
}

bool LLGAlign::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
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

	if(advancetime)
		spinto->time = spinfrom->time + spinfrom->dt;
	return true;
}









