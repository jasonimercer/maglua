// original code and calculations by Matthew Fitzpatrick 

#include <math.h>
#include "spinsystem.h" //so we can init with an ss to set size
#include "ewald3d.h"
#include <iostream>
using namespace std;

// helper function prototypes
static void cross(double* c, const double* a, const double* b);
static double dot3(const double* a, const double* b);
static void sum3(double* a, const double* b, const double* c);

// constants
#ifndef M_PI
#define M_PI 3.14159265358979323
#endif

#ifndef M_SQRTPI
#define M_SQRTPI 1.772453850905516027
#endif


// minimum search value for eta
#define MINA 1e-10

bool DipoleEwald3D::calcTensorElement(const int alpha, const int beta, const int* r3, double& element, const int nSites, const double tau)
{
	if(!gotGoodEta)
	{
		//		double r0[3] = {0,0,0};
		if(!findEta(a1, tau))
			return false;
	}

	double r[3];
	for(int i=0; i<3; i++)
	{
		r[i] = r3[0]*a1[i] + r3[1]*a2[i] + r3[2]*a3[i];
	}

	return calcTensorElementRealR(alpha, beta, r, element, nSites, tau);
}

bool DipoleEwald3D::calcTensorElementRealR(const int alpha, const int beta, const double* r, double& element, const int nSites, const double tau)
{
	if(!gotGoodEta)
	{
		if(!findEta(a1, tau))
			return false;
	}

	element = calcG_ab(alpha, beta, r, eta, nSites);
	return true;
}

void DipoleEwald3D::setUnitCell(const double* A1, const double* A2, const double* A3)
{
	for(int i=0; i<3; i++)
	{
		a1[i] = A1[i];
		a2[i] = A2[i];
		a3[i] = A3[i];
	}
	calcSupportVectors();
}

void DipoleEwald3D::setLatticeSiteCount(const int nx, const int ny, const int nz)
{
	d1 = nx;
	d2 = ny;
	d3 = nz;
	calcSupportVectors();
}

void DipoleEwald3D::calcSupportVectors()
{
	//cross products (only needed here)
	double u2Xu3[3], u3Xu1[3], u1Xu2[3]; 

	//Defining the super-lattice vectors
	for(int i=0; i<3; i++)
	{
		u1[i] = d1 * a1[i];
		u2[i] = d2 * a2[i];
		u3[i] = d3 * a3[i];
	}
	
	//Calculating cross-products
	cross(u2Xu3, u2, u3);
	cross(u3Xu1, u3, u1);
	cross(u1Xu2, u1, u2);

	//Calculating volume of 3D cell formed by {u1,u2,u3}
	vu = fabs(dot3(u1, u2Xu3));
	
	//Calculating reciprocal lattice vectors (triple product)
	for(int i=0; i<3; i++)
	{
		b1[i] = 2.0 * M_PI * u2Xu3[i] / vu;
		b2[i] = 2.0 * M_PI * u3Xu1[i] / vu;
		b3[i] = 2.0 * M_PI * u1Xu2[i] / vu;
	}
	gotGoodEta = false; //since we need to update based on new values
}



static int _findEtaDone(const double ca, const double tau, const double b, const double x)
{
	return  (ca) < (tau * (fabs(b) + fabs(x))); //then we've narrowed in on a good value
}

// http://en.wikipedia.org/wiki/Golden_section_search#Recursive_algorithm
// (1 + sqrt(5)) / 2
#define RGR 0.3819660112501051518
// recursive version of search with caching of function evaluations
bool DipoleEwald3D::_findEta(const int n, const double* r3,
							 const double a, const double fa,
							 const double b, const double fb,
							 const double c, const double fc,
							 const int nSites, const double tau)
{
	if(n < 0)
		return false;
	//printf("%g    %g    %g\n", a, b, c);
	if(a > MINA)
	{
		// checking to see if there is evidence for a minimum outside the bound
		if(fa < fb && fb < fc) //then lets try to extend the range to the right
		{
			double x = c + RGR*(c-b);
			const double fx = fabs(calcG_ab_nextTerm(0, 0, r3, x, nSites));
			return _findEta(n-1, r3, b, fb, c, fc, x, fx, nSites, tau);
		}
	}
	if(fa > fb && fb > fc) //then lets try to extend the range to the left
	{
		double x = a - RGR*(b-a);
		const double fx = fabs(calcG_ab_nextTerm(0, 0, r3, x, nSites));
		return _findEta(n-1, r3, x, fx, a, fa, b, fb, nSites, tau);
	}


	// selecting largest range for new probe point
	if((c-b) > (b-a)) //second range
	{
		const double x = b + (RGR) * (c-b);

		if(_findEtaDone(c-a, tau, b, x)) // comparing the size of the range (c-a) to the magnitude of the solution
		{
			eta = (c+a)*0.5;
			return true;
		}
		const double fx = fabs(calcG_ab_nextTerm(0, 0, r3, x, nSites));

		if(fx < fb)
		{
			return _findEta(n-1, r3, b, fb, x, fx, c, fc, nSites, tau);
		}
		else
		{
			return _findEta(n-1, r3, a, fa, b, fb, x, fx, nSites, tau);
		}
	}
	else // first range
	{
		const double x = b - RGR * (b-a);
		if(_findEtaDone(c-a, tau, b, x)) // comparing the size of the range (c-a) to the magnitude of the solution
		{
			eta = (c+a)*0.5;
			return true;
		}
		const double fx = fabs(calcG_ab_nextTerm(0, 0, r3, x, nSites));

		if(fx < fb)
		{
			return _findEta(n-1, r3, a, fa, x, fx, b, fb, nSites, tau);
		}
		else
		{
			return _findEta(n-1, r3, x, fx, b, fb, c, fc, nSites, tau);
		}
	}
}


bool DipoleEwald3D::findEta(const double* r3, const int nSites, double tau) //tau is like a tolerance parameter
{
	gotGoodEta = false;

	// required: c > a
	const double a = MINA;
	const double c = 1000;
	const double b = a + 0.3819660112501051518 * (c-a); //golden(ish) split
	
	const double fa =  fabs(calcG_ab_nextTerm(0, 0, r3, a, nSites));
	const double fb =  fabs(calcG_ab_nextTerm(0, 0, r3, b, nSites));
	const double fc =  fabs(calcG_ab_nextTerm(0, 0, r3, c, nSites));
	
	const int max_iterations = 100;
	
	//	FILE* f = fopen("d.dat", "w");
	//	for(double x=a; x<=2; x+=0.01)
	//	{
	//		const double fx = fabs(calcG_ab_nextTerm(0, 0, r3, x, nSites));
	//		fprintf(f, "%g %g\n", x, fx);
	//	}
	//	fclose(f);

	if(_findEta(max_iterations, r3, a, fa, b, fb, c, fc, nSites, tau))
	{
		//printf("Eta = %g\n", eta);
		gotGoodEta = true;
		return true;
	}
	return false;
}

//Function that calculates parts of individual terms of GAB
void DipoleEwald3D::calcG_ab_nmk(	const int alpha, const int beta,
								 const double* r, const double eta,
								 const int n, const int m, const int k,  //this is the part of the term requested
								 double& real_part, double& recip_part)
{
	if(dot3(r,r) == 0)
	{
		return calcG_ab_nmk0(alpha, beta, eta, n, m, k, real_part, recip_part);
	}
	//printf("!Zero\n");
	const double eta2  = eta*eta;
	const double delta = (alpha==beta)?1:0;

	double Q[3], rPlusR[3];

	// transform {n,m,k} into reciprocal and real space vectors
	Q[0] = n * b1[0] + m * b2[0] + k * b3[0];
	Q[1] = n * b1[1] + m * b2[1] + k * b3[1];
	Q[2] = n * b1[2] + m * b2[2] + k * b3[2];

	rPlusR[0] = n * u1[0] + m * u2[0] + k * u3[0] + r[0];
	rPlusR[1] = n * u1[1] + m * u2[1] + k * u3[1] + r[1];
	rPlusR[2] = n * u1[2] + m * u2[2] + k * u3[2] + r[2];

	//Fourier Space
	const double normQ2 = dot3(Q,Q);
	const double normQ1 = sqrt(normQ2);
	const double rDotQ = dot3(r, Q);
	//const double low_limit = 1e-5;
	const double low_limit = 0;

	if (normQ1 > low_limit) //factoring global 4.0 * M_PI / vu term in here. Not as efficient but reads easier on totalQ + totalR
		recip_part = 4.0 * M_PI * Q[alpha] * Q[beta] * cos(rDotQ) * exp(-normQ2 / (4.0 * eta2)) / (normQ2 * vu);
	else
	{
		recip_part = 0;
	}

	const double rRab = rPlusR[alpha] * rPlusR[beta];
	const double nrR2 = dot3(rPlusR, rPlusR);
	const double nrR1 = sqrt(nrR2);
	const double nrR4 = nrR2 * nrR2;
	const double nrR5 = nrR4 * nrR1;

	//Real Space
	if (nrR1 > low_limit)
	{
		real_part =
				2.0 * eta * exp(-eta2 * nrR2) * (nrR2 * delta - 3.0 * rRab - 2.0 * eta2 * rRab * nrR2) / (M_SQRTPI * nrR4) +
				(nrR2 * delta - 3.0 * rRab) * erfc (eta * nrR1) / (nrR5);
	}
	else
		real_part = 0;

}

//Function that calculates parts of individual terms of GAB - special case: r=0
void DipoleEwald3D::calcG_ab_nmk0(	const int alpha, const int beta,
								  const double eta,
								  const int n, const int m, const int k,  //this is the part of the term requested
								  double& real_part, double& recip_part)
{
	const double eta2  = eta*eta;
	const double delta = (alpha==beta)?1:0;
	
	double Q[3], rPlusR[3];

	// transform {n,m,k} into reciprocal and real space vectors
	Q[0] = n * b1[0] + m * b2[0] + k * b3[0];
	Q[1] = n * b1[1] + m * b2[1] + k * b3[1];
	Q[2] = n * b1[2] + m * b2[2] + k * b3[2];
	
	// r = 0
	rPlusR[0] = n * u1[0] + m * u2[0] + k * u3[0];
	rPlusR[1] = n * u1[1] + m * u2[1] + k * u3[1];
	rPlusR[2] = n * u1[2] + m * u2[2] + k * u3[2];
	
	//Fourier Space
	const double normQ2 = dot3(Q,Q);
	const double normQ1 = sqrt(normQ2);
	const double rDotQ = 0;
	const double low_limit = 0;

	if (normQ1 > low_limit) //factoring global 4.0 * M_PI / vu term in here. Not as efficient but reads easier on totalQ + totalR
	{
		recip_part = -4.0 * M_PI * exp(-normQ2 / (4.0 * eta2)) * (normQ2 * delta - 3 * Q[alpha] * Q[beta])  / (3.0 * vu * normQ2);
	}
	else
	{
		recip_part = 0;
	}

	const double rRab = rPlusR[alpha] * rPlusR[beta];
	const double nrR2 = dot3(rPlusR, rPlusR);
	const double nrR1 = sqrt(nrR2);
	const double nrR4 = nrR2 * nrR2;
	const double nrR5 = nrR4 * nrR1;
	
	//Real Space
	if (nrR1 > low_limit)
	{
		real_part = (nrR2 * delta - 3.0 * rPlusR[alpha] * rPlusR[beta]) *
				(2.0 * eta * exp(-nrR2 * eta2) * (3.0 + 2.0 * nrR2 * eta2)/(3.0 * nrR4 * M_SQRTPI) + erfc(nrR1 * eta) / nrR5);
	}
	else
		real_part = 0;
}

//Function that calculates individual terms of GAB
double DipoleEwald3D::calcG_ab(const int alpha, const int beta, const double* r, const double eta, const int nSites) 
{
	double totalQ = 0;
	double totalR = 0;
	
	double partQ, partR;

	for(int n=-nSites; n<=nSites; n++)
	{
		for (int m=-nSites; m<=nSites; m++)
		{
			for (int k=-nSites; k<=nSites; k++)
			{
				calcG_ab_nmk(alpha, beta, r, eta, n, m, k, partR, partQ);
				
				totalR += partR;
				totalQ += partQ;
			}
		}
	}

	return totalQ + totalR;
}

// Function that computes the contribution of the next term in the series (nSites + 1 term)
// this function is numerically equal to 
//   
//    calcG_ab(alpha, beta, r, eta, nSites+1) - calcG_ab(alpha, beta, r, eta, nSites)
//
// for infinite precision. At fixed precision (64 bit) the above goes to zero very quickly for good etas.
//
double DipoleEwald3D::calcG_ab_nextTerm(const int alpha, const int beta, const double* r, const double eta, const int nSites) 
{
	//  // for testing:
	// 	return calcG_ab(alpha, beta, r, eta, nSites+1) - calcG_ab(alpha, beta, r, eta, nSites);
	
	double totalQ = 0;
	double totalR = 0;
	
	double partQ1, partR1;
	double partQ2, partR2;

	// full k faces
	for(int n=-(nSites+1); n<=(nSites+1); n++)
	{
		for (int m=-(nSites+1); m<=(nSites+1); m++)
		{
			calcG_ab_nmk(alpha, beta, r, eta, n, m,  (nSites+1), partR1, partQ1);
			calcG_ab_nmk(alpha, beta, r, eta, n, m, -(nSites+1), partR2, partQ2);

			totalR += partR1;
			totalQ += partQ1;

			totalR += partR2;
			totalQ += partQ2;
		}
	}

	// m faces bounded by k
	for(int n=-(nSites+1); n<=(nSites+1); n++)
	{
		for (int k=-(nSites); k<=(nSites); k++)
		{
			calcG_ab_nmk(alpha, beta, r, eta, n,  (nSites+1), k, partR1, partQ1);
			calcG_ab_nmk(alpha, beta, r, eta, n, -(nSites+1), k, partR2, partQ2);

			totalR += partR1;
			totalQ += partQ1;

			totalR += partR2;
			totalQ += partQ2;
		}
	}

	// n faces bounded by k and m
	for(int m=-(nSites); m<=(nSites); m++)
	{
		for (int k=-(nSites); k<=(nSites); k++)
		{
			calcG_ab_nmk(alpha, beta, r, eta,  (nSites+1), m, k, partR1, partQ1);
			calcG_ab_nmk(alpha, beta, r, eta, -(nSites+1), m, k, partR2, partQ2);

			totalR += partR1;
			totalQ += partQ1;

			totalR += partR2;
			totalQ += partQ2;
		}
	}

	return totalQ + totalR;
}



// support functions
static void cross(double* c, const double* a, const double* b)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}
static double dot3(const double* a, const double* b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static double norm3(const double* a)
{
	return sqrt(dot3(a,a));
}
static void sum3(double* a, const double* b, const double* c)
{
	a[0] = b[0] + c[0];
	a[1] = b[1] + c[1];
	a[2] = b[2] + c[2];
}










// ===================================
//           lua functions
// ===================================
static int lua_getNint(lua_State* L, int N, int* vec, int pos, int def)
{
	if(lua_istable(L, pos))
	{
		for(int i=0; i<N; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, pos);
			if(lua_isnil(L, -1))
			{
				vec[i] = def;
			}
			else
			{
				vec[i] = lua_tointeger(L, -1);
			}
			lua_pop(L, 1);
		}
		return 1;
	}
	
	for(int i=0; i<N; i++)
	{
		if(lua_isnumber(L, pos+i))
		{
			vec[i] = lua_tointeger(L, pos+i);
		}
		else
		{
			vec[i] = def;
		}
	}
	
	return N;
}

static int lua_getNdouble(lua_State* L, int N, double* vec, int pos, double def)
{
	if(lua_istable(L, pos))
	{
		for(int i=0; i<N; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, pos);
			if(lua_isnil(L, -1))
			{
				vec[i] = def;
			}
			else
			{
				vec[i] = lua_tonumber(L, -1);
			}
			lua_pop(L, 1);
		}
		return 1;
	}
	
	for(int i=0; i<N; i++)
	{
		if(lua_isnumber(L, pos+i))
		{
			vec[i] = lua_tonumber(L, pos+i);
		}
		else
			return -1;
	}
	
	return N;
}



// LuaBaseObject methods:
int DipoleEwald3D::luaInit(lua_State* L)
{
	int n[3];
	
	if(luaT_is<SpinSystem>(L, 1))
	{
		SpinSystem* ss = luaT_to<SpinSystem>(L, 1);
		n[0] = ss->nx;
		n[1] = ss->ny;
		n[2] = ss->nz;
	}
	else
	{
		lua_getNint(L, 3, n, 1, 1);
	}

	for(int i=0; i<3; i++)
		if(n[i] <= 0)
			n[i] = 1;
	
	setLatticeSiteCount(n[0], n[1], n[2]);
	return 0;
}

void DipoleEwald3D::encode(buffer* b)
{
	char version = 0;
	encodeChar(version, b);
	
	encodeInteger(d1, b);
	encodeInteger(d2, b);
	encodeInteger(d3, b);
	
	for(int i=0; i<3; i++)
	{
		encodeDouble(a1[i], b);
		encodeDouble(a2[i], b);
		encodeDouble(a3[i], b);
	}
	
	encodeDouble(eta, b);
	encodeDouble(tau, b);
}
int  DipoleEwald3D::decode(buffer* b)
{
	char version = decodeChar(b);
	if(version == 0)
	{
		int nx = decodeInteger(b);
		int ny = decodeInteger(b);
		int nz = decodeInteger(b);
	
		for(int i=0; i<3; i++)
		{
			a1[i] = decodeDouble(b);
			a2[i] = decodeDouble(b);
			a3[i] = decodeDouble(b);
		}

		setLatticeSiteCount(nx, ny, nz); // calls 	calcSupportVectors();

		eta = decodeDouble(b);
		tau = decodeDouble(b);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
}


static int l_setunitcell(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);	
	
	int r1, r2, r3;
	
	double a[3], b[3], c[3];
	r1 = lua_getNdouble(L, 3, a, 2, 0);
	if(r1 < 0)	return luaL_error(L, "invalid vector");
	
	r2 = lua_getNdouble(L, 3, b, 2+r1, 0);
	if(r2 < 0)	return luaL_error(L, "invalid vector");

	r3 = lua_getNdouble(L, 3, c, 2+r1+r2, 0);
	if(r3 < 0)	return luaL_error(L, "invalid vector");

	if(dot3(a,a) < 0 || dot3(b,b) < 0 || dot3(c,c) < 0)
		return luaL_error(L, "zero vectors not allowed for set unit cell");
	
	e->setUnitCell(a,b,c);
	return 0;
}


static int l_getunitcell(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);	
	
	const double* x[3] = {e->a1, e->a2, e->a3};
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushnumber(L, x[i][j]);
			lua_settable(L, -3);
		}
	}
	return 3;
}

static int _abTo00(const char* AB, int* ab2)
{
	for(int i=0; i<2; i++)
	{
		switch(AB[i])
		{
		case 'x':
		case 'X':
			ab2[i] = 0;
			break;
		case 'y':
		case 'Y':
			ab2[i] = 1;
			break;
		case 'z':
		case 'Z':
			ab2[i] = 2;
			break;
		default:
			return 1;
		}
	}
	return 0;
}

static int l_calctens(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);

	const char* errmsg1 = "first argument of calculateTensorElement must be AB pair: `XX', `XY', ... `ZZ'";

	if(!lua_isstring(L, 2))
		return luaL_error(L, errmsg1);

	const char* AB = lua_tostring(L, 2);
	int ab[2];

	if(!AB || strlen(AB) < 2 || _abTo00(AB, ab))
		return luaL_error(L, errmsg1);

	int r[3];
	int r1 = lua_getNint(L, 3, r, 3, 0);

	double elem;
	bool ok = e->calcTensorElement(ab[0], ab[1], r, elem, e->getNSites(), e->getTau());

	if(!ok)
	{
		return luaL_error(L, "Failed to calculate real/reciprocal partition point (eta) for series limits. Try a different tau value for a tolerance (current tau = %f)", e->getTau());
	}

	lua_pushnumber(L, elem);
	return 1;
}

static int l_calctensrr(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);

	const char* errmsg1 = "first argument of calculateTensorElement must be AB pair: `XX', `XY', ... `ZZ'";

	if(!lua_isstring(L, 2))
		return luaL_error(L, errmsg1);

	const char* AB = lua_tostring(L, 2);
	int ab[2];

	if(!AB || strlen(AB) < 2 || _abTo00(AB, ab))
		return luaL_error(L, errmsg1);

	double r[3];
	int r1 = lua_getNdouble(L, 3, r, 3, 0);

	double elem;
	bool ok = e->calcTensorElementRealR(ab[0], ab[1], r, elem, e->getNSites(), e->getTau());

	if(!ok)
	{
		return luaL_error(L, "Failed to calculate real/reciprocal partition point (eta) for series limits. Try a different tau value for a tolerance (current tau = %f)", e->getTau());
	}

	lua_pushnumber(L, elem);
	return 1;
}

static int l_getucvol(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);
	lua_pushnumber(L, e->getVolume());
	return 1;
}

static int l_geteta(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);
	lua_pushnumber(L, e->getEta());
	return 1;
}
static int l_gettau(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);
	lua_pushnumber(L, e->getTau());
	return 1;
}
static int l_settau(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);
	e->setTau(lua_tonumber(L, 2));
	return 0;
}


static int l_getnsites(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);
	lua_pushnumber(L, e->getNSites());
	return 1;
}
static int l_setnsites(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);
	e->setNSites(lua_tointeger(L, 2));
	return 0;
}


static int l_getlatticesize(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);	
	lua_pushnumber(L, e->d1);
	lua_pushnumber(L, e->d2);
	lua_pushnumber(L, e->d3);
	return 3;
}

static int l_setlatticesize(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);	

	int n[3];
	
	if(luaT_is<SpinSystem>(L, 1))
	{
		SpinSystem* ss = luaT_to<SpinSystem>(L, 1);
		n[0] = ss->nx;
		n[1] = ss->ny;
		n[2] = ss->nz;
	}
	else
	{
		lua_getNint(L, 3, n, 1, 1);
	}

	for(int i=0; i<3; i++)
		if(n[i] <= 0)
			n[i] = 1;
	
	e->setLatticeSiteCount(n[0], n[1], n[2]);
	return 0;
}


int DipoleEwald3D::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Object used to calculate individual elements of the dipolar interaction tensors");
		lua_pushstring(L, "*3Vector* or *SpinSystem*: Optional lattice size"); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(!lua_isfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_setunitcell)
	{
		lua_pushstring(L, "Set the unit cell for the tensor calculations");
		lua_pushstring(L, "3 *3Vector*s: The a, b and c vectors making up the basis vectors for the unit cell,");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getunitcell)
	{
		lua_pushstring(L, "Get the unit cell for the tensor calculations");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 Tables: The a, b and c vectors making up the basis vectors for the unit cell,");
		return 3;
	}
	if(func == l_calctens)
	{
		lua_pushstring(L, "Calculate the tensor element.");
		lua_pushstring(L, "1 String, 1 *3Vector*: String is AB pair (`XX', `XY', ..., `ZZ') the *3Vector* represents offsets from zero. It takes 3 integer values each ranging from 0 to n-1 where n is the number of unit cells in the lattice in a given direction.");
		lua_pushstring(L, "1 Number: The tensor element at the requested position.");
		return 3;
	}
	if(func == l_calctensrr)
	{
		lua_pushstring(L, "Calculate the tensor element.");
		lua_pushstring(L, "1 String, 1 *3Vector*: String is AB pair (`XX', `XY', ..., `ZZ') the *3Vector* represents offsets from zero. It takes 3 real values representing a positon somewhere in the lattice. This 3D point is not transformed by the basis vectors.");
		lua_pushstring(L, "1 Number: The tensor element at the requested position.");
		return 3;
	}

	if(func == l_getlatticesize)
	{
		lua_pushstring(L, "Get the number of unit cells in each crystallographic direction.");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 Numbers: The number of unit cells in each crystallographic direction.");
		return 3;
	}
	if(func == l_setlatticesize)
	{
		lua_pushstring(L, "Set the number of unit cells in each crystallographic direction.");
		lua_pushstring(L, "1 *3Vector* or 1 *SpinSystem*: The number of unit cells in each crystallographic direction.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_gettau)
	{
		lua_pushstring(L, "Get the tolerance (tau) used in the search for a good eta value");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The new tolerance value (initially 0.001)");
		return 3;
	}
	if(func == l_settau)
	{
		lua_pushstring(L, "Set the tolerance (tau) used in the search for a good eta value");
		lua_pushstring(L, "1 Number: The new tolerance value (initially 0.001)");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_getnsites)
	{
		lua_pushstring(L, "Get the number of terms to use in the series along 1 direction in the 3D sum");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The number of terms (initially 10)");
		return 3;
	}
	if(func == l_setnsites)
	{
		lua_pushstring(L, "Set the number of terms to use in the series along 1 direction in the 3D sum");
		lua_pushstring(L, "1 Number: The number of terms (initially 10)");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_geteta)
	{
		lua_pushstring(L, "Get the crossover eta value determined from search.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The crossover value eta");
		return 3;
	}
	if(func == l_getucvol)
	{
		lua_pushstring(L, "Get the volume of the unit cell.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The volume.");
		return 3;
	}

	return LuaBaseObject::help(L);
}


static int l_tostring(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);	
	
	lua_pushfstring(L, "DipoleEwald3D (%dx%dx%d)", e->d1, e->d2, e->d3);
	return 1;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* DipoleEwald3D::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"__tostring",      l_tostring},
		{"setUnitCell",     l_setunitcell},
		{"unitCell",        l_getunitcell},
		{"volume",          l_getucvol},
		{"latticeSize",     l_getlatticesize},
		{"setLatticeSize",  l_setlatticesize},
		{"calculateTensorElement", l_calctens},
		{"calculateTensorElementRealR", l_calctensrr},

		{"tau", l_gettau},
		{"setTau", l_settau},

		{"NSites", l_getnsites},
		{"setNSites", l_setnsites},

		{"eta", l_geteta},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}





#include "info.h"
extern "C"
{
EWALD3D_API int lib_register(lua_State* L);
EWALD3D_API int lib_version(lua_State* L);
EWALD3D_API const char* lib_name(lua_State* L);
EWALD3D_API int lib_main(lua_State* L);
}


EWALD3D_API int lib_register(lua_State* L)
{
	luaT_register<DipoleEwald3D>(L);

	return 0;
}

EWALD3D_API int lib_version(lua_State* L)
{
	return __revi;
}

EWALD3D_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "DipoleEwald3D";
#else
	return "DipoleEwald3D-Debug";
#endif
}

EWALD3D_API int lib_main(lua_State* L)
{
	return 0;
}














#if 0
// testing
int main(int argc, char** argv)
{
	DipoleEwald3D e;

	double element;
	int r[3] = {0,1,0};

	e.calcTensorElement(0, 0, r, element);
	
	cout << element << endl;
	
	return 0;
}
#endif
