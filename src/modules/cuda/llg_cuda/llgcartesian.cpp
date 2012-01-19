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
	: LLG("Cartesian", ENCODE_LLGCART)
{
}


bool LLGCartesian::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
// bool LLGQuaternion::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
	// if new spins/fields exist on the host copy them to the device
	spinfrom->sync_spins_hd();
	spinto->sync_spins_hd();
	dmdt->sync_spins_hd();
	dmdt->sync_fields_hd(SUM_SLOT);
	
	const int nx = spinfrom->nx;
	const int ny = spinfrom->ny;
	const int nz = spinfrom->nz;
	
	const double gamma = dmdt->gamma;
	const double alpha = dmdt->alpha;
	const double dt    = dmdt->dt * scaledmdt;
	
#define S SUM_SLOT
	cuda_llg_cart_apply(nx, ny, nz,
						  spinto->d_x,   spinto->d_y,   spinto->d_z,   spinto->d_ms,
						spinfrom->d_x, spinfrom->d_y, spinfrom->d_z, spinfrom->d_ms,
						    dmdt->d_x,     dmdt->d_y,     dmdt->d_z,     dmdt->d_ms,
						    dmdt->d_hx[S], dmdt->d_hy[S], dmdt->d_hz[S],
						alpha, dt, gamma);	

	// mark spins as new for future d->h syncing
	spinto->new_device_spins = true;
	
	if(advancetime)
		spinto->time = spinfrom->time + dt;

	return true;
}










