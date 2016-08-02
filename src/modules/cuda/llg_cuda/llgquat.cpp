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
#include <stdlib.h>
#include <stdio.h>
#include "llgquat.h"
#include "spinsystem.h"
#include "spinoperation.h"

#include "llgquat.hpp"
#include "llg.h"

// Gilbert equation of motion:
//  dS    -g           a  g
//  -- = ----  S X h - ---- S X (S X h)
//  dt   1+aa          1+aa
//
// or
//
// dS    -g
// -- = ---- S X (h + a S X H)
// dt   1+aa
LLGQuaternion::LLGQuaternion()
	: LLG(hash32(LLGQuaternion::typeName()))
{
	registerWS();
}

LLGQuaternion::~LLGQuaternion()
{
	unregisterWS();
}

bool LLGQuaternion::apply(SpinSystem** spinfrom, double scaledmdt, SpinSystem** dmdt, SpinSystem** spinto, bool advancetime, int n)
{
	for(int i=0; i<n; i++)
	{
		apply(spinfrom[i], scaledmdt, dmdt[i], spinto[i], advancetime);
	}
}

bool LLGQuaternion::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
{
    // if new spins/fields exist on the host copy them to the device
//     dmdt->ensureSlotExists(SUM_SLOT);
//     dmdt->ensureSlotExists(THERMAL_SLOT);
	const int SUM_SLOT = dmdt->getSlot("Total");
	const int THERMAL_SLOT = dmdt->getSlot("Thermal");
	
	if(SUM_SLOT < 0) //nothing to do
	{
		if(advancetime)
			spinto->time = spinfrom->time + scaledmdt * dmdt->dt;
		return true;
	}

	const int nx = spinfrom->nx;
	const int ny = spinfrom->ny;
	const int nz = spinfrom->nz;
	
	const double gamma = dmdt->gamma;
	const double alpha = dmdt->alpha;
	const double dt    = dmdt->dt * scaledmdt;

	const double* d_gamma = dmdt->site_gamma?(dmdt->site_gamma->ddata()):0;
	const double* d_alpha = dmdt->site_alpha?(dmdt->site_alpha->ddata()):0;
	
	const int nxyz = nx*ny*nz;

	double* d_ws1;
	double* d_ws2;
	double* d_ws3;
	double* d_ws4;
	
	const int sz = sizeof(double)*nxyz;
	getWSMemD(&d_ws1, sz, hash32("SpinOperation::apply_1"));
	getWSMemD(&d_ws2, sz, hash32("SpinOperation::apply_2"));
	getWSMemD(&d_ws3, sz, hash32("SpinOperation::apply_3"));
	getWSMemD(&d_ws4, sz, hash32("SpinOperation::apply_4"));

#define S SUM_SLOT
#define T THERMAL_SLOT

#define dd(Q,xx)  ((Q>=0)?(xx?xx->ddata():0):0);

	double* Tx = dd(T,dmdt->hx[T]);
	double* Ty = dd(T,dmdt->hy[T]);
	double* Tz = dd(T,dmdt->hz[T]);
	
	double* Sx = dd(S,dmdt->hx[S]);
	double* Sy = dd(S,dmdt->hy[S]);
	double* Sz = dd(S,dmdt->hz[S]);

	double* stx = dd(1,spinto->x);
	double* sty = dd(1,spinto->y);
	double* stz = dd(1,spinto->z);
	double* stms= dd(1,spinto->ms);
	
	double* sfx = dd(1,spinfrom->x);
	double* sfy = dd(1,spinfrom->y);
	double* sfz = dd(1,spinfrom->z);
	double* sfms= dd(1,spinfrom->ms);
	
	double* dmx = dd(1,dmdt->x);
	double* dmy = dd(1,dmdt->y);
	double* dmz = dd(1,dmdt->z);
	double* dmms= dd(1,dmdt->ms);
	
	cuda_llg_quat_apply(nx, ny, nz,
			stx, sty, stz, stms,
			sfx, sfy, sfz, sfms,
			dmx, dmy, dmz, dmms,
            Tx, Ty, Tz,
            Sx, Sy, Sz,
            d_ws1, d_ws2, d_ws3, d_ws4,
			dt, alpha, d_alpha, gamma, d_gamma,
			thermalOnlyFirstTerm);	

	// mark spins as new for future d->h syncing
	spinto->x->new_device = true;
	spinto->y->new_device = true;
	spinto->z->new_device = true;
	
	if(advancetime)
		spinto->time = spinfrom->time + dt;

	return true;
}




int LLGQuaternion::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "LLG.Quaternion advances a *SpinSystem* through time using the Quaternion formulation of the LLG equation.");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return LLG::help(L);
}




