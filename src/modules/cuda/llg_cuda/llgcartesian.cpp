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
#include "llgcartesian.h"
#include "spinsystem.h"
#include "spinoperation.h"
#include "llgcartesian.hpp"

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
	: LLG(hash32(LLGCartesian::typeName()))
{
}

LLGCartesian::~LLGCartesian()
{
}

bool LLGCartesian::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
// bool LLGQuaternion::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
#define S SUM_SLOT
#define T THERMAL_SLOT
	dmdt->ensureSlotExists(SUM_SLOT);
	dmdt->ensureSlotExists(THERMAL_SLOT);


	const int nx = spinfrom->nx;
	const int ny = spinfrom->ny;
	const int nz = spinfrom->nz;
	
	const double gamma = dmdt->gamma;
	const double alpha = dmdt->alpha;
	const double dt    = dmdt->dt * scaledmdt;
	
	cuda_llg_cart_apply(nx, ny, nz,
			  spinto->x->ddata(),   spinto->y->ddata(),   spinto->z->ddata(),   spinto->ms->ddata(),
			spinfrom->x->ddata(), spinfrom->y->ddata(), spinfrom->z->ddata(), spinfrom->ms->ddata(),
			    dmdt->x->ddata(),     dmdt->y->ddata(),     dmdt->z->ddata(),     dmdt->ms->ddata(),
			    dmdt->hx[T]->ddata(), dmdt->hy[T]->ddata(), dmdt->hz[T]->ddata(),
			    dmdt->hx[S]->ddata(), dmdt->hy[S]->ddata(), dmdt->hz[S]->ddata(),
			alpha, dt, gamma);	
	
	spinto->x->new_device = true;
	spinto->y->new_device = true;
	spinto->z->new_device = true;
	spinto->ms->new_device = true;

	if(advancetime)
		spinto->time = spinfrom->time + dt;

	return true;
}


int LLGCartesian::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "LLG.Cartesian advances a *SpinSystem* through time using the Cartesian formulation of the LLG equation.");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return LLG::help(L);
}









