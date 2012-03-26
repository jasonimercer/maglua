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
	: LLG(hash32(LLGAlign::typeName()))
{
}

bool  LLGAlign::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
{
	const double* sx = spinfrom->x;
	const double* sy = spinfrom->y;
	const double* sz = spinfrom->z;
	const double* ms = spinfrom->ms;
	      double* mt = spinto->ms;

	const double* hx = dmdt->hx[SUM_SLOT];
	const double* hy = dmdt->hy[SUM_SLOT];
	const double* hz = dmdt->hz[SUM_SLOT];

	      double* x  = spinto->x;
	      double* y  = spinto->y;
	      double* z  = spinto->z;

	for(int i=0; i<spinfrom->nxyz; i++)
	{
		const double h = sqrt(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);
		mt[i] = ms[i];
		if(ms[i] > 0 && h > 0)
		{
			x[i] = ms[i] * hx[i] / h;
			y[i] = ms[i] * hy[i] / h;
			z[i] = ms[i] * hz[i] / h;
		}
	}

	if(advancetime)
		spinto->time = spinfrom->time + scaledmdt * dmdt->dt;
	return true;
}

int LLGAlign::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "LLG.Align dvances a *SpinSystem* through time aligning spins with local fields");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return LLG::help(L);
}








