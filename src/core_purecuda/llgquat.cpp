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
	: LLG("Quaternion", ENCODE_LLGQUAT)
{
}

bool LLGQuaternion::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
	const double gamma = spinfrom->gamma;
	const double alpha = spinfrom->alpha;
	const double dt    = spinfrom->dt;

	spinfrom->sync_spins_hd();
	fieldfrom->sync_fields_hd(SUM_SLOT);
	
	const int nx = spinfrom->nx;
	const int ny = spinfrom->ny;
	const int nz = spinfrom->nz;
	
	cuda_llg_quat_apply(nx, ny, nz,
						spinto->d_x, spinto->d_y, spinto->d_z, spinto->d_ms, 
						spinfrom->d_x, spinfrom->d_y, spinfrom->d_z, spinfrom->d_ms, 
								fieldfrom->d_hx[SUM_SLOT], 
								fieldfrom->d_hy[SUM_SLOT], 
								fieldfrom->d_hz[SUM_SLOT],
						spinfrom->d_ws1, spinfrom->d_ws2, spinfrom->d_ws3, spinfrom->d_ws4,
						alpha, dt, gamma);	
	
	spinto->new_device_spins = true;
	
	if(advancetime)
		spinto->time = spinfrom->time + dt;

	return true;
}








