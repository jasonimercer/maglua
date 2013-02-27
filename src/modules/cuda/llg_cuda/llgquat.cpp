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


bool LLGQuaternion::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
{
    // if new spins/fields exist on the host copy them to the device
    dmdt->ensureSlotExists(SUM_SLOT);
    dmdt->ensureSlotExists(THERMAL_SLOT);

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

#define dd(xx)  (xx?xx->ddata():0);

	double* Tx = dd(dmdt->hx[T]);
	double* Ty = dd(dmdt->hy[T]);
	double* Tz = dd(dmdt->hz[T]);
	
	double* Sx = dd(dmdt->hx[S]);
	double* Sy = dd(dmdt->hy[S]);
	double* Sz = dd(dmdt->hz[S]);

	double* stx = dd(spinto->x);
	double* sty = dd(spinto->y);
	double* stz = dd(spinto->z);
	double* stms= dd(spinto->ms);
	
	double* sfx = dd(spinfrom->x);
	double* sfy = dd(spinfrom->y);
	double* sfz = dd(spinfrom->z);
	double* sfms= dd(spinfrom->ms);
	
	double* dmx = dd(dmdt->x);
	double* dmy = dd(dmdt->y);
	double* dmz = dd(dmdt->z);
	double* dmms= dd(dmdt->ms);
	

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




