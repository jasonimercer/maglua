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
	registerWS();
}

LLGCartesian::~LLGCartesian()
{
	unregisterWS();
}


bool LLGCartesian::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
{
	SpinSystem* s1[1]; s1[0] = spinfrom;
	SpinSystem* s2[1]; s2[0] = dmdt;
	SpinSystem* s3[1]; s3[0] = spinto;
	return apply(s1, scaledmdt, s2, s3, advancetime, 1);
}

bool LLGCartesian::apply(SpinSystem** spinfrom, double scaledmdt, SpinSystem** dmdt, SpinSystem** spinto, bool advancetime, int n)
// bool LLGQuaternion::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
	vector<int> sum_slots;
	vector<int> thermal_slots;

	for(int i=0; i<n; i++)
	{
		sum_slots.push_back(dmdt[i]->getSlot("Total"));
		thermal_slots.push_back(dmdt[i]->getSlot("Thermal"));
	}
	
	const int nx = spinfrom[0]->nx;
	const int ny = spinfrom[0]->ny;
	const int nz = spinfrom[0]->nz;
	
	double** d_spinto_x_N = SpinOperation::getVectorOfVectors(spinto, n, "apply_1", 's', 'x');
	double** d_spinto_y_N = SpinOperation::getVectorOfVectors(spinto, n, "apply_2", 's', 'y');
	double** d_spinto_z_N = SpinOperation::getVectorOfVectors(spinto, n, "apply_3", 's', 'z');
	double** d_spinto_m_N = SpinOperation::getVectorOfVectors(spinto, n, "apply_4", 's', 'm');

	double** d_spinfrom_x_N = SpinOperation::getVectorOfVectors(spinfrom, n, "apply_5", 's', 'x');
	double** d_spinfrom_y_N = SpinOperation::getVectorOfVectors(spinfrom, n, "apply_6", 's', 'y');
	double** d_spinfrom_z_N = SpinOperation::getVectorOfVectors(spinfrom, n, "apply_7", 's', 'z');
	double** d_spinfrom_m_N = SpinOperation::getVectorOfVectors(spinfrom, n, "apply_8", 's', 'm');

	double** d_dmdt_x_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_9", 's', 'x');
	double** d_dmdt_y_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_10", 's', 'y');
	double** d_dmdt_z_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_11", 's', 'z');
	double** d_dmdt_m_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_12", 's', 'm');

	
	double** d_dmdt_hT_x_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_13", 'h', 'x', &(thermal_slots[0]));
	double** d_dmdt_hT_y_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_14", 'h', 'y', &(thermal_slots[0]));
	double** d_dmdt_hT_z_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_15", 'h', 'z', &(thermal_slots[0]));

	double** d_dmdt_hS_x_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_16", 'h', 'x', &(sum_slots[0]));
	double** d_dmdt_hS_y_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_17", 'h', 'y', &(sum_slots[0]));
	double** d_dmdt_hS_z_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_18", 'h', 'z', &(sum_slots[0]));

	double** d_alpha_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_19", 'a');
	double** d_gamma_N = SpinOperation::getVectorOfVectors(dmdt, n, "apply_20", 'g');

	double*  d_alpha   = (double*)SpinOperation::getVectorOfValues(dmdt, n, "apply_21", 'a');
	double*  d_gamma   = (double*)SpinOperation::getVectorOfValues(dmdt, n, "apply_22", 'g');
	double*  d_dt      = (double*)SpinOperation::getVectorOfValues(dmdt, n, "apply_23", 'd', scaledmdt);

	cuda_llg_cart_apply_N(nx, ny, nz,
			    d_spinto_x_N,   d_spinto_y_N,   d_spinto_z_N,   d_spinto_m_N,
			  d_spinfrom_x_N, d_spinfrom_y_N, d_spinfrom_z_N, d_spinfrom_m_N,
			      d_dmdt_x_N,     d_dmdt_y_N,     d_dmdt_z_N,     d_dmdt_m_N,
			      d_dmdt_hT_x_N,     d_dmdt_hT_y_N,     d_dmdt_hT_z_N,
			      d_dmdt_hS_x_N,     d_dmdt_hS_y_N,     d_dmdt_hS_z_N,
				d_dt, d_alpha_N, d_alpha, d_gamma_N, d_gamma, 
				thermalOnlyFirstTerm, disableRenormalization, n);	

// 	cuda_llg_cart_apply(nx, ny, nz,
// 			  spinto->x->ddata(),   spinto->y->ddata(),   spinto->z->ddata(),   spinto->ms->ddata(),
// 			spinfrom->x->ddata(), spinfrom->y->ddata(), spinfrom->z->ddata(), spinfrom->ms->ddata(),
// 			    dmdt->x->ddata(),     dmdt->y->ddata(),     dmdt->z->ddata(),     dmdt->ms->ddata(),
// 			    dmdt->hx[T]->ddata(), dmdt->hy[T]->ddata(), dmdt->hz[T]->ddata(),
// 			    dmdt->hx[S]->ddata(), dmdt->hy[S]->ddata(), dmdt->hz[S]->ddata(),
// 			dt, alpha, d_alpha, gamma, d_gamma, thermalOnlyFirstTerm, disableRenormalization);	


	for(int i=0; i<n; i++)
	{
		spinto[i]->x->new_device  = true;
		spinto[i]->y->new_device  = true;
		spinto[i]->z->new_device  = true;
		spinto[i]->ms->new_device = true;

		if(advancetime)
			spinto[i]->time = spinfrom[i]->time + spinfrom[i]->dt * scaledmdt;
	}
	
	

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









