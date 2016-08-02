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


    spinfrom->sync_spins_hd();
    spinto->sync_spins_hd();
    dmdt->sync_spins_hd();
    dmdt->sync_fields_hd(SUM_SLOT);

	const int nx = spinfrom->nx;
	const int ny = spinfrom->ny;
	const int nz = spinfrom->nz;
	
	const float gamma = dmdt->gamma;
	const float alpha = dmdt->alpha;
	const float dt    = dmdt->dt * scaledmdt;
	
	const int nxyz = nx*ny*nz;

	float* d_ws1;
	float* d_ws2;
	float* d_ws3;
	float* d_ws4;
	
	const int sz = sizeof(float)*nxyz;
	getWSMem(&d_ws1, sz, &d_ws2, sz, &d_ws3, sz, &d_ws4, sz);

#define S SUM_SLOT
#define T THERMAL_SLOT
	cuda_llg_quat_apply32(nx, ny, nz,
			  spinto->d_x,   spinto->d_y,   spinto->d_z,   spinto->d_ms,
			spinfrom->d_x, spinfrom->d_y, spinfrom->d_z, spinfrom->d_ms,
			    dmdt->d_x,     dmdt->d_y,     dmdt->d_z,     dmdt->d_ms,
            dmdt->d_hx[T], dmdt->d_hy[T], dmdt->d_hz[T],
			dmdt->d_hx[S], dmdt->d_hy[S], dmdt->d_hz[S],
			          d_ws1,         d_ws2,         d_ws3,         d_ws4,
			alpha, dt, gamma);	

	// mark spins as new for future d->h syncing
	spinto->new_device_spins = true;
	
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




