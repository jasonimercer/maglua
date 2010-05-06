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
	: LLG("Quaternion", ENCODE_LLGQUAT)
{
}




#define CROSS(v, a, b) \
	v[0] = a[1] * b[2] - a[2] * b[1]; \
	v[1] = a[2] * b[0] - a[0] * b[2]; \
	v[2] = a[0] * b[1] - a[1] * b[0];

bool LLGQuaternion::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
	const double* sx = spinfrom->x;
	const double* sy = spinfrom->y;
	const double* sz = spinfrom->z;
	const double* ms = spinfrom->ms;

	const double* hx = fieldfrom->hx[SUM_SLOT];
	const double* hy = fieldfrom->hy[SUM_SLOT];
	const double* hz = fieldfrom->hz[SUM_SLOT];

	      double* x  = spinto->x;
	      double* y  = spinto->y;
	      double* z  = spinto->z;
		  
	const double gamma = spinfrom->gamma;
	const double alpha = spinfrom->alpha;
	const double dt    = spinfrom->dt;

// dS    -g
// -- = ---- S X (h + a S X H)
// dt   1+aa
	Quaternion qRot;
	Quaternion qVec;
	Quaternion qRes;
	
	for(int i=0; i<spinfrom->nxyz; i++)
	{
		if(ms[i] > 0)
		{
			double HLen;
			double S[3];
			double H[3];
			double SH[3];
			double inv = 1.0 / ms[i];
			double ra; //rotate amount
			double sint, cost;
			
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

				double l = 1.0 / sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
				x[i] *= l * ms[i];
				y[i] *= l * ms[i];
				z[i] *= l * ms[i];
			}
			else
			{
				x[i] = sx[i];
				y[i] = sy[i];
				z[i] = sz[i];
			}

		}
	}

	if(advancetime)
		spinto->time = spinfrom->time + dt;

	return true;
}








