#include "wood_calculations.h"
#include <complex>
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979535
#endif
#define pi M_PI





/*
// dummy calculation to align with local field, ignoring anisotropy
int do_wood_calculation(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0)
{
	Cvctr Hunit = H.uVect();
	Cvctr Munit = M.uVect();
	
	if(Munit.DELTA(Hunit) < 0.5) //then close enough, don't care
	{
		Mout.set_cmp(M.x, M.y, M.z);
		return 0;
	}
	else //align with field
	{
		Mout.set_cmp(Hunit.x * M.mag, Hunit.y * M.mag, Hunit.z * M.mag);
		return 1; //made an update
	}
}
*/

int do_wood_calculation(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0)
{
	// "subscripts" u and q refer to the coordinates within the plane defined by the vectors K and H
	// the unit vector u points in the same direction as vector K
	// the normal of the plane is defined by (K cross H)/(Ks*Hs) = unit vector p
	// the unit vector q points perpendicular to both u and p. q = p X u = u X p X u

	complex<double> mu[4];			// the possible my components, in the new coordinate system, lying in the plane defined by K and H, the proper solution will be real
	double mq;					// q component of the magnetization
	double DELTA,DELTA_LOW;
	complex<double> fp,fn,e,d;			// the imaginary terms used to calculate my.
	double hu,hq,h;				// components of the scaled field vector Hs/Hk, where Hk = 2Ks/Ms
	double Hs,Ks,Ms;				// scalar values of the magnitudes of H,K, and M
	double cosa;					// cosine of the angle between H and K
	complex<double> ACOSE;			// the arc cosine for the e term
	complex<double> RADp,RADn;			// the different radical terms for my
	Cvctr MG[4];					// four possible solutions
	Cvctr MUV, MQV;				// two dummy vectors
	int scount;					// the solution count of non-imagninary solutions
	Cvctr u,q,p,ht;				// The unit vecotr defining the local coordinate system set by the anisotropy
	bool check;

	int n;						// Loop control
	Hs = H.mag;
	Ks = K.mag;
	Ms = M.mag;
// Define the local coordinate system
	u = K.uVect();
	ht = H.uVect();
	p = ht.cross(u);
	if (p.mag == 0.)
	   {q.set_cmp(1.,1.,1.);
		q = q.vadd(u);
		p = u.cross(q);}
	p = p.uVect();
	q = u.cross(p);
// scale h as a unitless quantity determined by magnetization, anisotripy, and the effective field.
	h = Hs/(2.0 * Ks/Ms);
	cosa = ht.dot(u);
// h components in terms of the local coordinate system
	hu = h*cosa;
	hq = sqrt(h*h - hu*hu);
// Analytic solutioins for magnetization minima/maxima formulated by Roger Wood
	d = 1.0 - h*h;
	if (d == 0.) d = 0.0000000000000000000000000000000000000001; // A cheat to avoid dividing by zero.
	ACOSE = 54.0*hq*hq*hu*hu/d/d/d - 1.0;
	ACOSE = ACOS_COMP(ACOSE);
	e    = d*cos( ACOSE/3.0 );
	fp   = +sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	fn   = -sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	RADp = sqrt(2.*fp*fp - 18.*e + 54.*hu*(1.+hq*hq)/fp );
	RADn = sqrt(2.*fn*fn - 18.*e + 54.*hu*(1.+hq*hq)/fn );
	mu[0] = fp/6.0 + RADp/6.0 - hu/2.0; // f and the radical sign are both positive
	mu[1] = fn/6.0 - RADn/6.0 - hu/2.0; // f and the radical sign are both negative
	mu[2] = fp/6.0 - RADp/6.0 - hu/2.0;
	mu[3] = fn/6.0 + RADn/6.0 - hu/2.0;
// Check the solutions and put good solutions onto a guess list
	n = 0;
	scount = 0;
	while (n < 4)
	{
		if ( ( abs( imag(mu[n]) )  < 1.e-16) && (abs(real(mu[n])) <= 1.0 + 0.01) )
		{
			mq = real(mu[n])*real(mu[n]);
			if (mq >= 1.0) mq = 0.0;
			else mq = sqrt( 1. -  mq);
			check = Stability(real(mu[n]),mq,hu,hq);
			if (check) 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MG[scount] = MUV;			
				MG[scount] = MG[scount].vadd(MQV);				// transformed.
				MG[scount].smult(Ms);
				scount ++;
			}
		}
		n++;
	}
	n = 1;
	if (scount == 0)   M0 = M;
	else M0 = MG[0];
	DELTA = M.DELTA(MG[0]);
	DELTA_LOW = DELTA;
//	go through the guess list and choose the solution that is closest to the original magnetization state
	while (n < scount)
	{
		DELTA = M.DELTA(MG[n]);
		if (DELTA < DELTA_LOW)
		{ DELTA_LOW = DELTA; M0 = MG[n]; }
		n++;
	}
	if ((M0.x == M.x) && (M0.y == M.y) && (M0.z = M.z))
	return 0;
	else return 1;
}

int do_wood_calculation_demag_min_close(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0, double DN)
{
	// "subscripts" u and q refer to the coordinates within the plane defined by the vectors K and H
	// the unit vector u points in the same direction as vector K
	// the normal of the plane is defined by (K cross H)/(Ks*Hs) = unit vector p
	// the unit vector q points perpendicular to both u and p. q = p X u = u X p X u
	complex<double> mu[4];				// the possible my components, in the new coordinate system, lying in the plane defined by K and H, the proper solution will be real
	double mq;				// q component of the magnetization
	double DELTA,DELTA_LOW;
	complex<double> fp,fn,e,d;			// the imaginary terms used to calculate my.
	double hu,hq,h;			// components of the scaled field vector Hs/Hk, where Hk = 2Ks/Ms
	double Hs,Ks,Ms;		// scalar values of the magnitudes of H,K, and M
	double cosa;			// cosine of the angle between H and K
	complex<double> ACOSE;				// the arc cosine for the e term
	complex<double> RADp,RADn;			// the different radical terms for my
	Cvctr MG[4];			// four possible solutions
	Cvctr MUV, MQV;			// two dummy vectors
	int scount;				// the solution count of non-imagninary solutions
	Cvctr u,q,p,ht;			// The unit vecotr defining the local coordinate system set by the anisotropy
	bool check;

	int n;					// Loop control
	Hs = H.mag;
	Ks = K.mag;
	Ms = M.mag;
// Define the local coordinate system
	u = K.uVect();
	ht = H.uVect();
	p = ht.cross(u);
	if (p.mag == 0.)
	{
		q.set_cmp(1.,1.,1.);
		q = q.vadd(u);
		p = u.cross(q);
	}
	p = p.uVect();
	q = u.cross(p);
// scale h as a unitless quantity determined by magnetization, anisotripy, and the effective field.
	h = Hs/(2.0 * Ks/Ms + DN*Ms);
	cosa = ht.dot(u);
// h components in terms of the local coordinate system
	hu = h*cosa;
	hq = sqrt(h*h - hu*hu);
// Analytic solutioins for magnetization minima/maxima formulated by Roger Wood
	d = 1.0 - h*h;
	if (d == 0.) d = 0.0000000000000000000000000000000000000001; // A cheat to avoid dividing by zero.
	ACOSE = 54.0*hq*hq*hu*hu/d/d/d - 1.0;
	ACOSE = ACOS_COMP(ACOSE);
	e    = d*cos( ACOSE/3.0 );
	fp   = +sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	fn   = -sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	RADp = sqrt(2.*fp*fp - 18.*e + 54.*hu*(1.+hq*hq)/fp );
	RADn = sqrt(2.*fn*fn - 18.*e + 54.*hu*(1.+hq*hq)/fn );
	mu[0] = fp/6.0 + RADp/6.0 - hu/2.0; // f and the radical sign are both positive
	mu[1] = fn/6.0 - RADn/6.0 - hu/2.0; // f and the radical sign are both negative
	mu[2] = fp/6.0 - RADp/6.0 - hu/2.0;
	mu[3] = fn/6.0 + RADn/6.0 - hu/2.0;
// Check the solutions and put good solutions onto a guess list
	n = 0;
	scount = 0;
	while (n < 4)
	{
		if ( ( abs( imag(mu[n]) )  < 1.e-16) && (abs(real(mu[n])) <= 1.0 + 0.01) )
		{
			mq = real(mu[n])*real(mu[n]);
			if (mq >= 1.0) mq = 0.0;
			else mq = sqrt( 1. -  mq);
			check = Stability(real(mu[n]),mq,hu,hq);
			if (check) 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MG[scount] = MUV;			
				MG[scount] = MG[scount].vadd(MQV);				// transformed.
				MG[scount].smult(Ms);
				scount ++;
			}
		}
		n++;
	}
	n = 1;
	if (scount == 0)   M0 = M;
	else M0 = MG[0];
	DELTA = M.DELTA(MG[0]);
	DELTA_LOW = DELTA;
//	go through the guess list and choose the solution that is closest to the original magnetization state
	while (n < scount)
	{
		DELTA = M.DELTA(MG[n]);
		if (DELTA < DELTA_LOW)
		{ DELTA_LOW = DELTA; M0 = MG[n]; }
		n++;
	}
	if ((M0.x == M.x) && (M0.y == M.y) && (M0.z = M.z))
	return 0;
	else return 1;
}

int do_wood_calculation_demag_min_far(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0, double DN)
{
	// "subscripts" u and q refer to the coordinates within the plane defined by the vectors K and H
	// the unit vector u points in the same direction as vector K
	// the normal of the plane is defined by (K cross H)/(Ks*Hs) = unit vector p
	// the unit vector q points perpendicular to both u and p. q = p X u = u X p X u
	complex<double> mu[4];				// the possible my components, in the new coordinate system, lying in the plane defined by K and H, the proper solution will be real
	double mq;				// q component of the magnetization
	double DELTA,DELTA_HIGH;
	complex<double> fp,fn,e,d;			// the imaginary terms used to calculate my.
	double hu,hq,h;			// components of the scaled field vector Hs/Hk, where Hk = 2Ks/Ms
	double Hs,Ks,Ms;		// scalar values of the magnitudes of H,K, and M
	double cosa;			// cosine of the angle between H and K
	complex<double> ACOSE;				// the arc cosine for the e term
	complex<double> RADp,RADn;			// the different radical terms for my
	Cvctr MG[4];			// four possible solutions
	Cvctr MUV, MQV;			// two dummy vectors
	int scount;				// the solution count of non-imagninary solutions
	Cvctr u,q,p,ht;			// The unit vecotr defining the local coordinate system set by the anisotropy
	bool check;

	int n;					// Loop control
	Hs = H.mag;
	Ks = K.mag;
	Ms = M.mag;
// Define the local coordinate system
	u = K.uVect();
	ht = H.uVect();
	p = ht.cross(u);
	if (p.mag == 0.)
	{
		q.set_cmp(1.,1.,1.);
		q = q.vadd(u);
		p = u.cross(q);
	}
	p = p.uVect();
	q = u.cross(p);
// scale h as a unitless quantity determined by magnetization, anisotripy, and the effective field.
	h = Hs/(2.0 * Ks/Ms + DN*Ms);
	cosa = ht.dot(u);
// h components in terms of the local coordinate system
	hu = h*cosa;
	hq = sqrt(h*h - hu*hu);
// Analytic solutioins for magnetization minima/maxima formulated by Roger Wood
	d = 1.0 - h*h;
	if (d == 0.) d = 0.0000000000000000000000000000000000000001; // A cheat to avoid dividing by zero.
	ACOSE = 54.0*hq*hq*hu*hu/d/d/d - 1.0;
	ACOSE = ACOS_COMP(ACOSE);
	e    = d*cos( ACOSE/3.0 );
	fp   = +sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	fn   = -sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	RADp = sqrt(2.*fp*fp - 18.*e + 54.*hu*(1.+hq*hq)/fp );
	RADn = sqrt(2.*fn*fn - 18.*e + 54.*hu*(1.+hq*hq)/fn );
	mu[0] = fp/6.0 + RADp/6.0 - hu/2.0; // f and the radical sign are both positive
	mu[1] = fn/6.0 - RADn/6.0 - hu/2.0; // f and the radical sign are both negative
	mu[2] = fp/6.0 - RADp/6.0 - hu/2.0;
	mu[3] = fn/6.0 + RADn/6.0 - hu/2.0;
// Check the solutions and put good solutions onto a guess list
	n = 0;
	scount = 0;
	while (n < 4)
	{
		if ( ( abs( imag(mu[n]) )  < 1.e-16) && (abs(real(mu[n])) <= 1.0 + 0.01) )
		{
			mq = real(mu[n])*real(mu[n]);
			if (mq >= 1.0) mq = 0.0;
			else mq = sqrt( 1. -  mq);
			check = Stability(real(mu[n]),mq,hu,hq);
			if (check) 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MG[scount] = MUV;			
				MG[scount] = MG[scount].vadd(MQV);				// transformed.
				MG[scount].smult(Ms);
				scount ++;
			}
		}
		n++;
	}
	n = 1;
	if (scount == 0)   M0 = M;
	else M0 = MG[0];
	DELTA = M.DELTA(MG[0]);
	DELTA_HIGH = DELTA;
//	go through the guess list and choose the solution that is furthest from the original magnetization state
	while (n < scount)
	{
		DELTA = M.DELTA(MG[n]);
		if (DELTA > DELTA_HIGH)
		{ DELTA_HIGH = DELTA; M0 = MG[n]; }
		n++;
	}
	if ((M0.x == M.x) && (M0.y == M.y) && (M0.z = M.z))
	return 0;
	else return 1;
}

int do_wood_calculation_demag_max(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0, double DN)
{
	// "subscripts" u and q refer to the coordinates within the plane defined by the vectors K and H
	// the unit vector u points in the same direction as vector K
	// the normal of the plane is defined by (K cross H)/(Ks*Hs) = unit vector p
	// the unit vector q points perpendicular to both u and p. q = p X u = u X p X u
	complex<double> mu[4];				// the possible my components, in the new coordinate system, lying in the plane defined by K and H, the proper solution will be real
	double mq;				// q component of the magnetization
	double DELTA,DELTA_LOW;
	complex<double> fp,fn,e,d;			// the imaginary terms used to calculate my.
	double hu,hq,h;			// components of the scaled field vector Hs/Hk, where Hk = 2Ks/Ms
	double Hs,Ks,Ms;		// scalar values of the magnitudes of H,K, and M
	double cosa;			// cosine of the angle between H and K
	complex<double> ACOSE;				// the arc cosine for the e term
	complex<double> RADp,RADn;			// the different radical terms for my
	Cvctr MG[4];			// four possible solutions
	Cvctr MUV, MQV;			// two dummy vectors
	int scount;				// the solution count of non-imagninary solutions
	Cvctr u,q,p,ht;			// The unit vecotr defining the local coordinate system set by the anisotropy
	bool check;

	int n;					// Loop control
	Hs = H.mag;
	Ks = K.mag;
	Ms = M.mag;
// Define the local coordinate system
	u = K.uVect();
	ht = H.uVect();
	p = ht.cross(u);
	if (p.mag == 0.)
	{
		q.set_cmp(1.,1.,1.);
		q = q.vadd(u);
		p = u.cross(q);
	}
	p = p.uVect();
	q = u.cross(p);
// scale h as a unitless quantity determined by magnetization, anisotripy, and the effective field.
	h = Hs/(2.0 * Ks/Ms + DN*Ms);
	cosa = ht.dot(u);
// h components in terms of the local coordinate system
	hu = h*cosa;
	hq = sqrt(h*h - hu*hu);
// Analytic solutioins for magnetization minima/maxima formulated by Roger Wood
	d = 1.0 - h*h;
	if (d == 0.) d = 0.0000000000000000000000000000000000000001; // A cheat to avoid dividing by zero.
	ACOSE = 54.0*hq*hq*hu*hu/d/d/d - 1.0;
	ACOSE = ACOS_COMP(ACOSE);
	e    = d*cos( ACOSE/3.0 );
	fp   = +sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	fn   = -sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	RADp = sqrt(2.*fp*fp - 18.*e + 54.*hu*(1.+hq*hq)/fp );
	RADn = sqrt(2.*fn*fn - 18.*e + 54.*hu*(1.+hq*hq)/fn );
	mu[0] = fp/6.0 + RADp/6.0 - hu/2.0; // f and the radical sign are both positive
	mu[1] = fn/6.0 - RADn/6.0 - hu/2.0; // f and the radical sign are both negative
	mu[2] = fp/6.0 - RADp/6.0 - hu/2.0;
	mu[3] = fn/6.0 + RADn/6.0 - hu/2.0;
// Check the solutions and put good solutions onto a guess list
	n = 0;
	scount = 0;
	while (n < 4)
	{
		if ( ( abs( imag(mu[n]) )  < 1.e-16) && (abs(real(mu[n])) <= 1.0 + 0.01) )
		{
			mq = real(mu[n])*real(mu[n]);
			if (mq >= 1.0) mq = 0.0;
			else mq = sqrt( 1. -  mq);
			check = Stability(real(mu[n]),mq,hu,hq);
			if (!check) 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MG[scount] = MUV;			
				MG[scount] = MG[scount].vadd(MQV);				// transformed.
				MG[scount].smult(Ms);
				scount ++;
			}
		}
		n++;
	}
	n = 1;
	if (scount == 0)   M0 = M;
	else M0 = MG[0];
	DELTA = M.DELTA(MG[0]);
	DELTA_LOW = DELTA;
//	go through the guess list and choose the solution that is closest to the original magnetization state
	while (n < scount)
	{
		DELTA = M.DELTA(MG[n]);
		if (DELTA < DELTA_LOW)
		{ DELTA_LOW = DELTA; M0 = MG[n]; }
		n++;
	}
	if ((M0.x == M.x) && (M0.y == M.y) && (M0.z = M.z))
	return 0;
	else return 1;
}

int EffAni(Cvctr& K, Cvctr M, double Nx, double Ny, double Nz)
{
	Cvctr A;
	double alf, phi, beta;
	double K0, A0;
	double Ms;
	double D,lmb;
	double r;
	double denom;

	K0 = K.mag;
	Ms = M.mag;
	D = Nx*Ms;
	lmb = (Nz - Nx)/Nx;
	r = D/K0 * lmb;
	alf = acos(K.z/K0);
	if (K.x != 0) phi = atan(abs(K.y/K.x));
	else phi = pi;
	if (K.x <  0.0 && K.y >= 0.0) phi = pi-phi;
	if (K.x <  0.0 && K.y <  0.0) phi = pi+phi;
	if (K.x >= 0.0 && K.y <  0.0) phi = 2.0*pi-phi;
	A0 = K0*( 1 + r* ( cos(2.0*alf)+ r) / ( r + cos(alf)*cos(alf) ) );
	denom = 1.0 + r*cos(alf)*cos(alf)-r*(1+cos(2.0*alf))+r*r;
	beta = acos( (cos(alf)*cos(alf)-r)/sqrt(denom) );
	K.set_ang(A0,beta,phi);
	return 1;
}

int EffAni2(Cvctr& K, Cvctr M, double Nx, double Ny, double Nz)
// the values passed to this fuction should be an anisotropy axis, a magnetization
// and the converted (to the appropriate units) demagnetizing factors from a magnetostatic matrix
// The demagnetizing factors should multiplied by the correct conversion factor
// for example, the demagnetizing factors nxx, nyy, and nzz should all add to 1
// this would make the converted demagnetizing factors:
//		Nx = 4*pi*nxx, Ny = 4*pi*nyy, and Nz = 4*pi*nzz	--in cgs units
//		Ny = nxx, nyy, nzz	-- in SI units
{
	Cvctr A;
	double alf, phi, beta;
	double K0, A0;
	double Ms;
	double D,lmb;
	double r;
	double denom;

	K0 = K.mag;
	Ms = M.mag;
	D = abs(Nx*Ms*Ms);
	lmb = (Nx - Nz)/Nx;
	r = D/K0 * lmb;
	alf = acos(K.z/K0);
	if (K.x != 0) phi = atan(abs(K.y/K.x));
	else phi = pi;
	if (K.x <  0.0 && K.y >= 0.0) phi = pi-phi;
	if (K.x <  0.0 && K.y <  0.0) phi = pi+phi;
	if (K.x >= 0.0 && K.y <  0.0) phi = 2.0*pi-phi;
	A0 = K0*( 1.0 + r * (cos(2.0*alf)+r)/(cos(alf)*cos(alf)+r) );
	denom = 1.0 + r * (cos(2.0*alf)+r)/(cos(alf)*cos(alf)+r) ;
	beta = (cos(alf)*cos(alf) + r)/ denom;
	beta = sqrt(beta);
	beta = acos( beta );
	K.set_ang(A0,beta,phi);
	return 1;
}

double DU(Cvctr H, Cvctr K, Cvctr M, double DN) // return the smallest energy density barrier between the current minimum and the other minimum
{
	// "subscripts" u and q refer to the coordinates within the plane defined by the vectors K and H
	// the unit vector u points in the same direction as vector K
	// the normal of the plane is defined by (K cross H)/(Ks*Hs) = unit vector p
	// the unit vector q points perpendicular to both u and p. q = p X u = u X p X u
	complex<double> mu[4];				// the possible my components, in the new coordinate system, lying in the plane defined by K and H, the proper solution will be real
	double mq;				// q component of the magnetization
	double DELTA,DELTA_LOW;
	complex<double> fp,fn,e,d;			// the imaginary terms used to calculate my.
	double hu,hq,h;			// components of the scaled field vector Hs/Hk, where Hk = 2Ks/Ms
	double Hs,Ks,Ms;		// scalar values of the magnitudes of H,K, and M
	double cosa;			// cosine of the angle between H and K
	complex<double> ACOSE;				// the arc cosine for the e term
	complex<double> RADp,RADn;			// the different radical terms for my
	Cvctr MUV, MQV;			// two dummy vectors
// -- Tools for calculating the energy barrier
	Cvctr MAX[4];			// maximums
	Cvctr MIN[4];			// minimums
	Cvctr Mmin;				// The minimum for the given orientation
	double DELTA_U;				// The solution to be returned.
	double Umin,Umax;		// The energy densities for the chosen minimums and maximums
	int smin, smax;			// the solution count of non-imagninary solutions
	int imin, imax;			// the indices for the chosen minimum and maximum
	Cvctr u,q,p,ht;			// The unit vecotr defining the local coordinate system set by the anisotropy

	bool check;

	int n;					// Loop control
	Hs = H.mag;
	Ks = K.mag;
	Ms = M.mag;
// Define the local coordinate system
	u = K.uVect();
	ht = H.uVect();
	p = ht.cross(u);
	if (p.mag == 0.)
	   {q.set_cmp(1.,1.,1.);
		q = q.vadd(u);
		p = u.cross(q);}
	p = p.uVect();
	q = u.cross(p);
// scale h as a unitless quantity determined by magnetization, anisotripy, and the effective field.
	h = Hs/(2.0 * Ks/Ms + DN*Ms);
	cosa = ht.dot(u);
// h components in terms of the local coordinate system
	hu = h*cosa;
	hq = sqrt(h*h - hu*hu);
// Analytic solutioins for magnetization minima/maxima formulated by Roger Wood
	d = 1.0 - h*h;
	if (d == 0.) d = 0.0000000000000000000000000000000000000001; // A cheat to avoid dividing by zero.
	ACOSE = 54.0*hq*hq*hu*hu/d/d/d - 1.0;
	ACOSE = ACOS_COMP(ACOSE);
	e    = d*cos( ACOSE/3.0 );
	fp   = +sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	fn   = -sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	RADp = sqrt(2.*fp*fp - 18.*e + 54.*hu*(1.+hq*hq)/fp );
	RADn = sqrt(2.*fn*fn - 18.*e + 54.*hu*(1.+hq*hq)/fn );
	mu[0] = fp/6.0 + RADp/6.0 - hu/2.0; // f and the radical sign are both positive
	mu[1] = fn/6.0 - RADn/6.0 - hu/2.0; // f and the radical sign are both negative
	mu[2] = fp/6.0 - RADp/6.0 - hu/2.0;
	mu[3] = fn/6.0 + RADn/6.0 - hu/2.0;
// Check the solutions and put good solutions onto a guess list
	n = 0;
	smax = 0;
	smin = 0;
	for (n = 0; n < 4; n++)
	{
		if ( ( abs( imag(mu[n]) )  < 1.e-16) && (abs(real(mu[n])) <= 1.0 + 0.01) )
		{
			mq = real(mu[n])*real(mu[n]);
			if (mq >= 1.0) mq = 0.0;
			else mq = sqrt( 1. -  mq);
			check = Stability(real(mu[n]),mq,hu,hq);
			if (check) 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MIN[smin] = MUV;			
				MIN[smin] = MIN[smin].vadd(MQV);				// transformed.
				MIN[smin].smult(Ms);
				smin ++;
			}
			else 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MAX[smax] = MUV;			
				MAX[smax] = MAX[smax].vadd(MQV);				// transformed.
				MAX[smax].smult(Ms);
				smax ++;
			}
		}
	}
	Mmin = MIN[0];
	DELTA = M.DELTA(MIN[0]);
	DELTA_LOW = DELTA;
//	go through the minimum list and choose the result that is closest to the original magnetization state
	for (n = 1; n < smin; n++)
	{
		DELTA = M.DELTA(MIN[n]);
		if (DELTA < DELTA_LOW)
		{ DELTA_LOW = DELTA; Mmin = MIN[n]; }
		n++;
	}
// find the differnce between energy densities of the minimum state and the maximum states and choose the small difference

	u.set_cmp(0,0,1);
	double M2cos2_min,M2cos2_max;
	M2cos2_min = u.dot(Mmin)*u.dot(Mmin);
	M2cos2_max = u.dot(MAX[0])*u.dot(MAX[0]);
	Umin =  -(K.dot(Mmin  )/Mmin.mag  )*(K.dot(Mmin  )/Mmin.mag  )/K.mag - H.dot(Mmin  )-(0.5)*DN*M2cos2_min;
	Umax =  -(K.dot(MAX[0])/MAX[0].mag)*(K.dot(MAX[0])/MAX[0].mag)/K.mag - H.dot(MAX[0])-(0.5)*DN*M2cos2_max;

	DELTA_U = Umax - Umin;
//	cout << endl;
//	cout << "Maximum #1 = " << MAX[0].x << ", " << MAX[0].y << ", " << MAX[0].z << endl;
	for (n =1; n < smax; n++)
	{
//		cout << "Maximum #2 = " << MAX[n].x << ", " << MAX[n].y << ", " << MAX[n].z << endl;
		M2cos2_max = u.dot(MAX[n])*u.dot(MAX[n]);
		Umax = -(K.dot(MAX[n])/MAX[n].mag)*(K.dot(MAX[n])/MAX[n].mag)/K.mag - H.dot(MAX[n])-(0.5)*DN*M2cos2_max;
		if (Umax - Umin < DELTA_U) DELTA_U = Umax - Umin;
	}
 	if (smin <= 1) DELTA_U = 1.0e20;
	return DELTA_U;
}

double DU_return(Cvctr H, Cvctr K, Cvctr M, double DN) // return the smallest energy density barrier between the furthest minimum and the current minimum
{
	// "subscripts" u and q refer to the coordinates within the plane defined by the vectors K and H
	// the unit vector u points in the same direction as vector K
	// the normal of the plane is defined by (K cross H)/(Ks*Hs) = unit vector p
	// the unit vector q points perpendicular to both u and p. q = p X u = u X p X u
	complex<double> mu[4];				// the possible my components, in the new coordinate system, lying in the plane defined by K and H, the proper solution will be real
	double mq;				// q component of the magnetization
	double DELTA,DELTA_HIGH;
	complex<double> fp,fn,e,d;			// the imaginary terms used to calculate my.
	double hu,hq,h;			// components of the scaled field vector Hs/Hk, where Hk = 2Ks/Ms
	double Hs,Ks,Ms;		// scalar values of the magnitudes of H,K, and M
	double cosa;			// cosine of the angle between H and K
	complex<double> ACOSE;				// the arc cosine for the e term
	complex<double> RADp,RADn;			// the different radical terms for my
	Cvctr MUV, MQV;			// two dummy vectors
// -- Tools for calculating the energy barrier
	Cvctr MAX[4];			// maximums
	Cvctr MIN[4];			// minimums
	Cvctr Mmin;				// The minimum for the given orientation
	double DELTA_U;				// The solution to be returned.
	double Umin,Umax;		// The energy densities for the chosen minimums and maximums
	int smin, smax;			// the solution count of non-imagninary solutions
	int imin, imax;			// the indices for the chosen minimum and maximum
	Cvctr u,q,p,ht;			// The unit vecotr defining the local coordinate system set by the anisotropy

	bool check;

	int n;					// Loop control
	Hs = H.mag;
	Ks = K.mag;
	Ms = M.mag;
// Define the local coordinate system
	u = K.uVect();
	ht = H.uVect();
	p = ht.cross(u);
	if (p.mag == 0.)
	   {q.set_cmp(1.,1.,1.);
		q = q.vadd(u);
		p = u.cross(q);}
	p = p.uVect();
	q = u.cross(p);
// scale h as a unitless quantity determined by magnetization, anisotripy, and the effective field.
	h = Hs/(2.0 * Ks/Ms + DN*Ms);
	cosa = ht.dot(u);
// h components in terms of the local coordinate system
	hu = h*cosa;
	hq = sqrt(h*h - hu*hu);
// Analytic solutioins for magnetization minima/maxima formulated by Roger Wood
	d = 1.0 - h*h;
	if (d == 0.) d = 0.0000000000000000000000000000000000000001; // A cheat to avoid dividing by zero.
	ACOSE = 54.0*hq*hq*hu*hu/d/d/d - 1.0;
	ACOSE = ACOS_COMP(ACOSE);
	e    = d*cos( ACOSE/3.0 );
	fp   = +sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	fn   = -sqrt(9.0*hu*hu + 6.0*d + 6.0*e);
	RADp = sqrt(2.*fp*fp - 18.*e + 54.*hu*(1.+hq*hq)/fp );
	RADn = sqrt(2.*fn*fn - 18.*e + 54.*hu*(1.+hq*hq)/fn );
	mu[0] = fp/6.0 + RADp/6.0 - hu/2.0; // f and the radical sign are both positive
	mu[1] = fn/6.0 - RADn/6.0 - hu/2.0; // f and the radical sign are both negative
	mu[2] = fp/6.0 - RADp/6.0 - hu/2.0;
	mu[3] = fn/6.0 + RADn/6.0 - hu/2.0;
// Check the solutions and put good solutions onto a guess list
	n = 0;
	smax = 0;
	smin = 0;
	for (n = 0; n < 4; n++)
	{
		if ( ( abs( imag(mu[n]) )  < 1.e-16) && (abs(real(mu[n])) <= 1.0 + 0.01) )
		{
			mq = real(mu[n])*real(mu[n]);
			if (mq >= 1.0) mq = 0.0;
			else mq = sqrt( 1. -  mq);
			check = Stability(real(mu[n]),mq,hu,hq);
			if (check) 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MIN[smin] = MUV;			
				MIN[smin] = MIN[smin].vadd(MQV);				// transformed.
				MIN[smin].smult(Ms);
				smin ++;
			}
			else 
			{
				MUV = u; MQV = q;
				MUV.smult(real(mu[n]));	// begin transforming the solution in the local coordinates
				MQV.smult(mq);			// back in the global coordinates
				MAX[smax] = MUV;			
				MAX[smax] = MAX[smax].vadd(MQV);				// transformed.
				MAX[smax].smult(Ms);
				smax ++;
			}
		}
	}
	Mmin = MIN[0];
	DELTA = M.DELTA(MIN[0]);
	DELTA_HIGH = DELTA;
//	go through the minimum list and choose the result that is furthest to the original magnetization state
	for (n = 1; n < smin; n++)
	{
		DELTA = M.DELTA(MIN[n]);
		if (DELTA > DELTA_HIGH)
		{ DELTA_HIGH = DELTA; Mmin = MIN[n]; }
		n++;
	}
// find the differnce between energy densities of the minimum state and the maximum states and choose the small difference

	u.set_cmp(0,0,1);
	double M2cos2_min,M2cos2_max;
	M2cos2_min = u.dot(Mmin)*u.dot(Mmin);
	M2cos2_max = u.dot(MAX[0])*u.dot(MAX[0]);
	Umin =  -(K.dot(Mmin  )/Mmin.mag  )*(K.dot(Mmin  )/Mmin.mag  )/K.mag - H.dot(Mmin  )-(0.5)*DN*M2cos2_min;
	Umax =  -(K.dot(MAX[0])/MAX[0].mag)*(K.dot(MAX[0])/MAX[0].mag)/K.mag - H.dot(MAX[0])-(0.5)*DN*M2cos2_max;

	DELTA_U = Umax - Umin;
//	cout << endl;
//	cout << "Maximum #1 = " << MAX[0].x << ", " << MAX[0].y << ", " << MAX[0].z << endl;
	for (n =1; n < smax; n++)
	{
//		cout << "Maximum #2 = " << MAX[n].x << ", " << MAX[n].y << ", " << MAX[n].z << endl;
		M2cos2_max = u.dot(MAX[n])*u.dot(MAX[n]);
		Umax = -(K.dot(MAX[n])/MAX[n].mag)*(K.dot(MAX[n])/MAX[n].mag)/K.mag - H.dot(MAX[n])-(0.5)*DN*M2cos2_max;
		if (Umax - Umin < DELTA_U) DELTA_U = Umax - Umin;
	}
// 	if (smin <= 1) DELTA_U = 1.0e20;
	return DELTA_U;
}


bool Stability(double mu, double mq, double hu, double hq)
{
	double ddE;
	bool check;
	ddE =  2.*(mu*mu - mq*mq); 
	ddE += 2.*(hu*mu + hq*mq);
	if (ddE >  0.0) check = true;
	if (ddE <= 0.0) check = false;
	return(check);
}

complex<double> ACOS_COMP(complex<double> Z) // Returns a complex result for the arc cosine of a complex number.
{
	double X,Y;
	double A,B;
	complex<double> ACOS_C;
	complex<double> ci(0,1);
	X = real(Z);
	Y = imag(Z);
	A  = 0.5*sqrt((X+1)*(X+1) + Y*Y);
	A += 0.5*sqrt((X-1)*(X-1) + Y*Y);
	B  = 0.5*sqrt((X+1)*(X+1) + Y*Y);
	B -= 0.5*sqrt((X-1)*(X-1) + Y*Y);
	if (Y >= 0.0) ACOS_C = acos(B) - ci*log( A+sqrt(A*A-1.0) );
	if (Y <  0.0) ACOS_C = acos(B) + ci*log( A+sqrt(A*A-1.0) );
	ACOS_C = acos(B) - ci*log(A+sqrt(A*A-1.0));
	ACOS_C = acos(B) + ci*log(A+sqrt(A*A-1.0));
	return( ACOS_C );
}
