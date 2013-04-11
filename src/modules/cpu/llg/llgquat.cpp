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

#include "llg.h"

typedef struct{double w, x, y, z;} Quaternion;

static Quaternion qmult(Quaternion a, Quaternion b)
{
	Quaternion ab;

	ab.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
	ab.x = a.w*b.x + b.w*a.x  + a.y*b.z - a.z*b.y;
	ab.y = a.w*b.y + b.w*a.y  + a.z*b.x - a.x*b.z;
	ab.z = a.w*b.z + b.w*a.z  + a.x*b.y - a.y*b.x;

	return ab;
}

///Quaternion multiply without calculating the W component
static Quaternion qmultXYZ(Quaternion a, Quaternion b) 
{
	Quaternion ab;

	ab.x = a.w*b.x + b.w*a.x  + a.y*b.z - a.z*b.y;
	ab.y = a.w*b.y + b.w*a.y  + a.z*b.x - a.x*b.z;
	ab.z = a.w*b.z + b.w*a.z  + a.x*b.y - a.y*b.x;
	ab.w = 0;

	return ab;
}

static Quaternion qconjugate(Quaternion q)
{
	Quaternion qq;
	qq.w = q.w;
	qq.x = -1.0 * q.x;
	qq.y = -1.0 * q.y;
	qq.z = -1.0 * q.z;
	return qq;
}



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
}


#define CROSS(v, a, b) \
	v[0] = a[1] * b[2] - a[2] * b[1]; \
	v[1] = a[2] * b[0] - a[0] * b[2]; \
	v[2] = a[0] * b[1] - a[1] * b[0];

class gamma_alpha
{
public:
	gamma_alpha(SpinSystem* ss)
	{
		// making local refcounted references to data so the
		// arrays don't get free'd in mid-operation (from another thread)
		site_alpha = luaT_inc<dArray>(ss->site_alpha);
		site_gamma = luaT_inc<dArray>(ss->site_gamma);
		
		a_alpha = (site_alpha)?ss->site_alpha->data():0;
		a_gamma = (site_gamma)?ss->site_gamma->data():0;
		
		v_alpha = ss->alpha;
		v_gamma = ss->gamma;
	}
	
	~gamma_alpha()
	{
		luaT_dec<dArray>(site_alpha);
		luaT_dec<dArray>(site_gamma);
	}
	
	double gamma(int idx)
	{
		if(a_gamma)
			return a_gamma[idx];
		return v_gamma;
	}
	double alpha(int idx)
	{
		if(a_alpha)
			return a_alpha[idx];
		return v_alpha;
	}
	
	dArray* site_alpha;
	dArray* site_gamma;
	
	double* a_gamma;
	double* a_alpha;
	
	double v_gamma;
	double v_alpha;
};

bool LLGQuaternion::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
{
	const int SUM_SLOT = dmdt->getSlot("Total");
	const int THERMAL_SLOT = dmdt->getSlot("Thermal");
	
	if(SUM_SLOT < 0) //nothing to do
	{
		if(advancetime)
			spinto->time = spinfrom->time + dmdt->dt * scaledmdt;
		return true;
	}
	
	const double* sx = spinfrom->x->data();
	const double* sy = spinfrom->y->data();
	const double* sz = spinfrom->z->data();
	const double* ms = spinfrom->ms->data();

	      double* mt = spinto->ms->data();

	const double* hx = dmdt->hx[SUM_SLOT]->data();
	const double* hy = dmdt->hy[SUM_SLOT]->data();
	const double* hz = dmdt->hz[SUM_SLOT]->data();

	const double* mx = dmdt->x->data();
	const double* my = dmdt->y->data();
	const double* mz = dmdt->z->data();

	      double* x  = spinto->x->data();
	      double* y  = spinto->y->data();
	      double* z  = spinto->z->data();
		  
	gamma_alpha ga(dmdt);
// 	const double gamma = dmdt->gamma;
// 	const double alpha = dmdt->alpha;
	const double dt    = dmdt->dt * scaledmdt;

// dS    -g
// -- = ---- S X (h + a S X H)
// dt   1+aa
	
// 	#pragma omp parallel for private (qRot, qVec, qRes) shared(hx, hy, hz, sx, sy, sz, x, y, z)
	const int nxyz = spinfrom->nxyz;
	if(thermalOnlyFirstTerm)
	{
// 		#pragma omp parallel for shared(x, y, z)
		for(int i=0; i<nxyz; i++)
		{
			Quaternion qRot;
			Quaternion qVec;
			Quaternion qRes;
			
			mt[i] = ms[i];
			if(ms[i] > 0)
			{
				double HLen;
				double S[3];
				double H[3];
				double h[3];
				double M[3];
				double Sh[3];
				const double inv = 1.0 / ms[i];
				double ra; //rotate amount
				double sint, cost;
				
				const double alpha = ga.alpha(i);
				const double gamma = ga.gamma(i);
				
				// dS    -g                  a
				// -- = ---- S X ((H+Hth) + ---S X H)
				// dt   1+aa                |S|

				// here the thermal field is bundled up in Heff
				// we need to subtract out that contribution for the
				// second term of the LLG equation

				S[0]=sx[i]; S[1]=sy[i]; S[2]=sz[i];
				M[0]=mx[i]; M[1]=my[i]; M[2]=mz[i];
				H[0]=hx[i]; H[1]=hy[i]; H[2]=hz[i];

				if(THERMAL_SLOT >= 0)
				{
					h[0] = H[0] - dmdt->hx[THERMAL_SLOT]->data()[i];
					h[1] = H[1] - dmdt->hy[THERMAL_SLOT]->data()[i];
					h[2] = H[2] - dmdt->hz[THERMAL_SLOT]->data()[i];
				}
				else
				{
					h[0] = H[0];
					h[1] = H[1];
					h[2] = H[2];
				}
				
				CROSS(Sh, M, h);
				for(int j=0; j<3; j++)
					H[j] += alpha * Sh[j] * inv;


				HLen = sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]);

				qVec.w = 0;
				qVec.x = S[0]; //rotate this vector
				qVec.y = S[1];
				qVec.z = S[2];
				
				if(HLen > 0)
				{
					double iHLen = 1.0 / HLen;
					ra = 1.0 * gamma * HLen / (1.0 + alpha * alpha);

					cost = cos(0.5 * ra * dt);
					sint = sin(0.5 * ra * dt);
					
	// 				printf("theta: %g\n", 0.5 * ra * dt);

					qRot.w = cost;
					qRot.x = sint * H[0] * iHLen; //rotate about damped h
					qRot.y = sint * H[1] * iHLen;
					qRot.z = sint * H[2] * iHLen;

					//this is the rotation: qRes = qRot qVec qRot*
					qRes = qmultXYZ(qmult(qRot, qVec), qconjugate(qRot));

					x[i] = qRes.x;
					y[i] = qRes.y;
					z[i] = qRes.z;

// 					const double xyz = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
// 					if(xyz > 0)
// 					{
// 						const double l = 1.0 / sqrt(xyz);
// 						x[i] *= l * ms[i];
// 						y[i] *= l * ms[i];
// 						z[i] *= l * ms[i];
// 					}
				}
				else
				{
					x[i] = sx[i];
					y[i] = sy[i];
					z[i] = sz[i];
				}

			}
		}
	}
	else
	{
// 		#pragma omp parallel for shared(x, y, z)
		for(int i=0; i<spinfrom->nxyz; i++)
		{
			Quaternion qRot;
			Quaternion qVec;
			Quaternion qRes;
			
			mt[i] = ms[i];
			if(ms[i] > 0)
			{
				double HLen;
				double S[3];
				double H[3];
				double SH[3];
				const double inv = 1.0 / ms[i];
				double ra; //rotate amount
				double sint, cost;
				
				const double alpha = ga.alpha(i);
				const double gamma = ga.gamma(i);
				
				// dS    -g           a
				// -- = ---- S X (H +---S X H)
				// dt   1+aa         |S|

				S[0]=sx[i]; S[1]=sy[i]; S[2]=sz[i];
				H[0]=hx[i]; H[1]=hy[i]; H[2]=hz[i];

				CROSS(SH, S, H);
				for(int j=0; j<3; j++)
					H[j] += alpha * SH[j] * inv;

				HLen = sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]);

				qVec.w = 0;
				qVec.x = S[0]; //rotate this vector
				qVec.y = S[1];
				qVec.z = S[2];
				
				if(HLen > 0)
				{
					double iHLen = 1.0 / HLen;
					ra = 1.0 * gamma * HLen / (1.0 + alpha * alpha);

					cost = cos(0.5 * ra * dt);
					sint = sin(0.5 * ra * dt);
					
	// 				printf("theta: %g\n", 0.5 * ra * dt);

					qRot.w = cost;
					qRot.x = sint * H[0] * iHLen; //rotate about damped h
					qRot.y = sint * H[1] * iHLen;
					qRot.z = sint * H[2] * iHLen;

					//this is the rotation: qRes = qRot qVec qRot*
					qRes = qmultXYZ(qmult(qRot, qVec), qconjugate(qRot));

					x[i] = qRes.x;
					y[i] = qRes.y;
					z[i] = qRes.z;

// 					const double xyz = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
// 					if(xyz > 0)
// 					{
// 						const double l = 1.0 / sqrt(xyz);
// 						x[i] *= l * ms[i];
// 						y[i] *= l * ms[i];
// 						z[i] *= l * ms[i];
// 					}
				}
				else
				{
					x[i] = sx[i];
					y[i] = sy[i];
					z[i] = sz[i];
				}

			}
		}
		
	}
	
	if(advancetime)
		spinto->time = spinfrom->time + dt;

	if(!disableRenormalization)
	{
		for(int i=0; i<nxyz; i++)
		{
			const double mm = x[i]*x[i] + y[i]*y[i] + z[i]*z[i];
			if(mm != 0)
			{
				const double inv = ms[i] / sqrt(mm);

				x[i] *= inv;
				y[i] *= inv;
				z[i] *= inv;
			}
		}
	}
	
	
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






