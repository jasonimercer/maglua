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
	SpinSystem** s1[1]; s1[0] = spinfrom;
	SpinSystem** s2[1]; s2[0] = dmdt;
	SpinSystem** s3[1]; s3[0] = spinto;
	return apply(s1, scaledmdt, s2, s3, advancetime);
}

bool LLGCartesian::apply(SpinSystem** spinfrom, double scaledmdt, SpinSystem** dmdt, SpinSystem** spinto, bool advancetime, int n)
// bool LLGQuaternion::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
#define S SUM_SLOT
#define T THERMAL_SLOT
	for(int i=0; i<n; i++)
	{
		dmdt[i]->ensureSlotExists(SUM_SLOT);
		dmdt[i]->ensureSlotExists(THERMAL_SLOT);
	}
	
	const int nx = spinfrom[0]->nx;
	const int ny = spinfrom[0]->ny;
	const int nz = spinfrom[0]->nz;
	
	const double **d_spinto_x_N,   **d_spinto_y_N,   **d_spinto_z_N,   **d_spinto_m_N;
	const double **h_spinto_x_N,   **h_spinto_y_N,   **h_spinto_z_N,   **h_spinto_m_N;

	const double **d_spinfrom_x_N, **d_spinfrom_y_N, **d_spinfrom_z_N, **d_spinfrom_m_N;
	const double **h_spinfrom_x_N, **h_spinfrom_y_N, **h_spinfrom_z_N, **h_spinfrom_m_N;

	const double **d_dmdt_x_N,     **d_dmdt_y_N,     **d_dmdt_z_N,     **d_dmdt_m_N;
	const double **h_dmdt_x_N,     **h_dmdt_y_N,     **h_dmdt_z_N,     **h_dmdt_m_N;

	getWSMemD(&d_spinto_x_N, sizeof(double*)*n, hash32("SpinOperation::apply_1"));
	getWSMemD(&d_spinto_y_N, sizeof(double*)*n, hash32("SpinOperation::apply_2"));
	getWSMemD(&d_spinto_z_N, sizeof(double*)*n, hash32("SpinOperation::apply_3"));
	getWSMemD(&d_spinto_m_N, sizeof(double*)*n, hash32("SpinOperation::apply_4"));

	getWSMemH(&h_spinto_x_N, sizeof(double*)*n, hash32("SpinOperation::apply_1"));
	getWSMemH(&h_spinto_y_N, sizeof(double*)*n, hash32("SpinOperation::apply_2"));
	getWSMemH(&h_spinto_z_N, sizeof(double*)*n, hash32("SpinOperation::apply_3"));
	getWSMemH(&h_spinto_m_N, sizeof(double*)*n, hash32("SpinOperation::apply_4"));

	
	
	getWSMemD(&d_spinfrom_x_N, sizeof(double*)*n, hash32("SpinOperation::apply_5"));
	getWSMemD(&d_spinfrom_y_N, sizeof(double*)*n, hash32("SpinOperation::apply_6"));
	getWSMemD(&d_spinfrom_z_N, sizeof(double*)*n, hash32("SpinOperation::apply_7"));
	getWSMemD(&d_spinfrom_m_N, sizeof(double*)*n, hash32("SpinOperation::apply_8"));

	getWSMemH(&h_spinfrom_x_N, sizeof(double*)*n, hash32("SpinOperation::apply_5"));
	getWSMemH(&h_spinfrom_y_N, sizeof(double*)*n, hash32("SpinOperation::apply_6"));
	getWSMemH(&h_spinfrom_z_N, sizeof(double*)*n, hash32("SpinOperation::apply_7"));
	getWSMemH(&h_spinfrom_m_N, sizeof(double*)*n, hash32("SpinOperation::apply_8"));
	
	
	
	getWSMemD(&d_dmdt_x_N, sizeof(double*)*n, hash32("SpinOperation::apply_9"));
	getWSMemD(&d_dmdt_y_N, sizeof(double*)*n, hash32("SpinOperation::apply_10"));
	getWSMemD(&d_dmdt_z_N, sizeof(double*)*n, hash32("SpinOperation::apply_11"));
	getWSMemD(&d_dmdt_m_N, sizeof(double*)*n, hash32("SpinOperation::apply_12"));

	getWSMemH(&h_dmdt_x_N, sizeof(double*)*n, hash32("SpinOperation::apply_9"));
	getWSMemH(&h_dmdt_y_N, sizeof(double*)*n, hash32("SpinOperation::apply_10"));
	getWSMemH(&h_dmdt_z_N, sizeof(double*)*n, hash32("SpinOperation::apply_11"));
	getWSMemH(&h_dmdt_m_N, sizeof(double*)*n, hash32("SpinOperation::apply_12"));

	
	
	getWSMemD(&d_hx_N, sizeof(double*)*n, hash32("SpinOperation::apply_4"));
	getWSMemD(&d_hy_N, sizeof(double*)*n, hash32("SpinOperation::apply_5"));
	getWSMemD(&d_hz_N, sizeof(double*)*n, hash32("SpinOperation::apply_6"));
	
	getWSMemH(&h_hx_N, sizeof(double*)*n, hash32("SpinOperation::apply_4"));
	getWSMemH(&h_hy_N, sizeof(double*)*n, hash32("SpinOperation::apply_5"));
	getWSMemH(&h_hz_N, sizeof(double*)*n, hash32("SpinOperation::apply_6"));
	

	const double** d_spinto_x_N = new const double*[n];
	const double** d_spinto_y_N = new const double*[n];
	const double** d_spinto_z_N = new const double*[n];
	const double** d_spinto_m_N = new const double*[n];
		
	const double** d_spinfrom_x_N = new const double*[n];
	const double** d_spinfrom_y_N = new const double*[n];
	const double** d_spinfrom_z_N = new const double*[n];
	const double** d_spinfrom_m_N = new const double*[n];

	const double** d_dmdt_x_N = new const double*[n];
	const double** d_dmdt_y_N = new const double*[n];
	const double** d_dmdt_z_N = new const double*[n];
	const double** d_dmdt_m_N = new const double*[n];
	
	const double** d_dmdt_hT_x_N = new const double*[n];
	const double** d_dmdt_hT_y_N = new const double*[n];
	const double** d_dmdt_hT_z_N = new const double*[n];

	const double** d_dmdt_hS_x_N = new const double*[n];
	const double** d_dmdt_hS_y_N = new const double*[n];
	const double** d_dmdt_hS_z_N = new const double*[n];

	double* d_gamma;
	double* h_gamma;

	double* d_alpha;
	double* h_alpha;

	double* d_dt;
	double* h_dt;

	getWSMemD(&d_gamma, sizeof(double)*n, hash32("SpinOperation::apply_1"));
	getWSMemH(&h_gamma, sizeof(double)*n, hash32("SpinOperation::apply_1"));

	getWSMemD(&d_alpha, sizeof(double)*n, hash32("SpinOperation::apply_2"));
	getWSMemH(&h_alpha, sizeof(double)*n, hash32("SpinOperation::apply_2"));

	getWSMemD(&d_dt,    sizeof(double)*n, hash32("SpinOperation::apply_3"));
	getWSMemH(&h_dt,    sizeof(double)*n, hash32("SpinOperation::apply_3"));

	for(int i=0; i<n; i++)
	{
		h_gamma[i] = dmdt[i]->gamma;
		h_alpha[i] = dmdt[i]->alpha;
		h_dt[i]    = dmdt[i]->dt * scaledmdt;
	}

	memcpy_h2d(d_gamma, h_gamma, sizeof(double)*n);
	memcpy_h2d(d_alpha, h_alpha, sizeof(double)*n);
	memcpy_h2d(d_dt,    h_dt,    sizeof(double)*n);


	const double* d_gamma = dmdt->site_gamma?(dmdt->site_gamma->ddata()):0;
	const double* d_alpha = dmdt->site_alpha?(dmdt->site_alpha->ddata()):0;

	const double** d_gamma_N = new const double*[n];
	const double** d_alpha_N = new const double*[n];
	
	
	for(int i=0; i<n; i++)
	{
		d_spinto_x_N[i] = spinto[i]->x->ddata();
		d_spinto_y_N[i] = spinto[i]->y->ddata();
		d_spinto_z_N[i] = spinto[i]->z->ddata();
		d_spinto_m_N[i] = spinto[i]->ms->ddata();

		d_spinfrom_x_N[i] = spinfrom[i]->x->ddata();
		d_spinfrom_y_N[i] = spinfrom[i]->y->ddata();
		d_spinfrom_z_N[i] = spinfrom[i]->z->ddata();
		d_spinfrom_m_N[i] = spinfrom[i]->ms->ddata();
		
		d_dmdt_x_N[i] = dmdt[i]->x->ddata();
		d_dmdt_y_N[i] = dmdt[i]->y->ddata();
		d_dmdt_z_N[i] = dmdt[i]->z->ddata();
		d_dmdt_m_N[i] = dmdt[i]->ms->ddata();
				
		d_dmdt_hT_x_N[i] = dmdt[i]->hx[T]->ddata();
		d_dmdt_hT_y_N[i] = dmdt[i]->hy[T]->ddata();
		d_dmdt_hT_z_N[i] = dmdt[i]->hz[T]->ddata();

		d_dmdt_hS_x_N[i] = dmdt[i]->hx[S]->ddata();
		d_dmdt_hS_y_N[i] = dmdt[i]->hy[S]->ddata();
		d_dmdt_hS_z_N[i] = dmdt[i]->hz[S]->ddata();

	}
	
	
	cuda_llg_cart_apply_N(nx, ny, nz,
			    d_spinto_x_N,   d_spinto_y_N,   d_spinto_z_N,   d_spinto_m_N,
			  d_spinfrom_x_N, d_spinfrom_y_N, d_spinfrom_z_N, d_spinfrom_m_N,
			      d_dmdt_x_N,     d_dmdt_y_N,     d_dmdt_z_N,     d_dmdt_m_N,
			      d_dmdt_hT_x_N,     d_dmdt_hT_y_N,     d_dmdt_hT_z_N,
			      d_dmdt_hS_x_N,     d_dmdt_hS_y_N,     d_dmdt_hS_z_N,

// 			  dmdt->x->ddata(),     dmdt->y->ddata(),     dmdt->z->ddata(),     dmdt->ms->ddata(),
			    dmdt->hx[T]->ddata(), dmdt->hy[T]->ddata(), dmdt->hz[T]->ddata(),
			    dmdt->hx[S]->ddata(), dmdt->hy[S]->ddata(), dmdt->hz[S]->ddata(),
			dt, alpha, d_alpha, gamma, d_gamma, thermalOnlyFirstTerm, disableRenormalization);	

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
			spinto->time = spinfrom->time + dt;
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









