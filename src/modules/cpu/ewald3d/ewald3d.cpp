// original code and calculations by Matthew Fitzpatrick 

#include <math.h>
#include <string.h> //for memcpy
#include "spinsystem.h" //so we can init with an ss to set size
#include "ewald3d.h"



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


bool DipoleEwald3D::calcTensorElement(const int alpha, const int beta, const int* r3, double& element, double tau = 0.001)
{
	if(!gotGoodEta)
	{
		if(!findEta(tau))
			return false;
	}
	
	double r[3];
	for(int i=0; i<3; i++)
	{
		r[i] = r3[0]*a1[i] + r3[1]*a2[i] + r3[2]*a3[i];
	}
	
	element = calcG_ab(alpha, beta, r, eta, nSites);
	return true;
}

	
void DipoleEwald3D::setUnitCell(const double* A1, const double* A2, const double* A3)
{
	memcpy(a1, A1, sizeof(double)*3);
	memcpy(a2, A2, sizeof(double)*3);
	memcpy(a3, A3, sizeof(double)*3);
	calcSupportVectors();
}

void DipoleEwald3D::setUnitCell(const vector<double>& A1, const vector<double>& A2, const vector<double>& A3)
{
	setUnitCell(&A1[0], &A2[0], &A3[0]);
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
	vu = dot3(u1, u2Xu3);
	
	//Calculating reciprocal lattice vectors (triple product)
	for(int i=0; i<3; i++)
	{
		b1[i] = 2.0 * M_PI * u2Xu3[i] / vu;
		b2[i] = 2.0 * M_PI * u3Xu1[i] / vu;
		b3[i] = 2.0 * M_PI * u1Xu2[i] / vu;
	}
	gotGoodEta = false; //since we need to update based on new values
}

bool DipoleEwald3D::findEta(double tau) //tau is like a tolerance parameter
{
	gotGoodEta = false;
	//Beginning of Golden Section Search to find best convergence parameter eta
	//Variables for golden section search of best convergence parameter eta
	int iterations = 0;
	double phi = (sqrt(5.0) - 1.0) / 2.0; //Golden ratio
	double x1 = 1.0e-10;
	double x3 = 5.;
	double x2 = x1 + (1 - phi) * (x3 - x1);
	double x4;
	double error;
	
	const double* r = a1;

	double leftBracket, rightBracket, mid;
	while(iterations <= 100)
	{
		mid = (x3 + x1) / 2.0; //x1 and x3 are the left and right brackets respectively of the section we are currently searching
// 		error = fabs(calcG_ab(0, 0, r, mid, nSites)-calcG_ab(0, 0, r, mid, (nSites+1))); //Find the error at mid-point of section
		error = fabs(calcG_ab_nextTerm(0, 0, r, mid, nSites)); //Find the error at mid-point of section
		leftBracket = x1;
		rightBracket = x3;

		if (fabs(x3 - x1) < tau * (fabs(x2) + fabs(x4))) //Check to see if search section is small enough 
		{
// 			cout << "a=" << x1 << "\t b=" << x3 << "\t" << "eta = " << mid << ", diff = " << error << "\n \n";
			break;
		}
		
		if ((x3-x2) > (x2-x1)) //Else we must shrink our search window based on Golden ratio
		{
			x4 = x2 + (1.0 - phi) * (x3 - x2);
		}
		else
		{
			x4 = x1 + phi * (x2 - x1);
		}

		//Calculate differences in dipole sums for different numbers of terms (nSites vs. (nSites+1))
		delGXXx2 = fabs(calcG_ab_nextTerm(0, 0, r, x2, nSites));
		delGXXx4 = fabs(calcG_ab_nextTerm(0, 0, r, x4, nSites));
		delGXXleft = fabs(calcG_ab_nextTerm(0, 0, r, leftBracket, nSites));
		delGXXright = fabs(calcG_ab_nextTerm(0, 0, r, rightBracket, nSites));
// 		cout << leftBracket << "\t" << rightBracket << "\t" << x2 << "\t" << x4 << "\t" << delGXXleft << "\t" << delGXXright << "\t" << "\t" << delGXXx2 << "\t" << delGXXx4 << "\n";
// 		cout << iterations << "\t" << mid << "\t" << error << endl;
		if (delGXXx4 < delGXXx2)
		{
			if ((x3 - x2) > (x2 - x1))
			{
				x1 = x2;
				x2 = x4;
			}
			else
			{
				x3 = x2;
				x2 = x4;
			}
		}
		else
		{
			if ((x3 - x2) > (x2 - x1))
			{
				x3 = x4;
			}
			else
			{
				x1 = x4;
			}
		}
		iterations++;
	}
	if (iterations == 101) //Not a good enough eta was found
	{
// 		cout << "Did not find a good convergence parameter after " << iterations-1 << " iterations; \n";
// 		cout << "Program Aborted. \n";
		return false;
	}

	eta = mid; //Set last midpoint in search algorithm to our eta	
	gotGoodEta = true;
	return true;
}


 //Function that calculates parts of individual terms of GAB
void DipoleEwald3D::calcG_ab_nmk(	const int alpha, const int beta, 
									const double* r, const double eta, 
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
	
	rPlusR[0] = n * u1[0] + m * u2[0] + k * u3[0] + r[0];
	rPlusR[1] = n * u1[1] + m * u2[1] + k * u3[1] + r[1];
	rPlusR[2] = n * u1[2] + m * u2[2] + k * u3[2] + r[2];
	
	//Fourier Space
	const double normQ2 = dot3(Q,Q);
	const double normQ1 = sqrt(normQ2);
	const double rDotQ = dot3(r, Q);
	const double low_limit = 1e-8;
	if (normQ1 >= low_limit) //factoring global 4.0 * M_PI / vu term in here. Not as efficient but reads easier on totalQ + totalR
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
	if (nrR1 >= low_limit)
	{
		real_part = 
			2.0 * eta * exp(-eta2 * nrR2) * (nrR2 * delta - 3.0 * rRab - 2.0 * eta2 * rRab * nrR2) / (M_SQRTPI * nrR4) +
			(nrR2 * delta - 3.0 * rRab) * erfc (eta * nrR1) / (nrR5);
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

//	printf("r = {%g %g %g}\n", r[0], r[1], r[2]);

/*
	printf("a1  %g %g %g\n", a1[0], a1[1], a1[2]);
	printf("a2  %g %g %g\n", a2[0], a2[1], a2[2]);
	printf("a3  %g %g %g\n", a3[0], a3[1], a3[2]);

	printf("b1  %g %g %g\n", b1[0], b1[1], b1[2]);
	printf("b2  %g %g %g\n", b2[0], b2[1], b2[2]);
	printf("b3  %g %g %g\n", b3[0], b3[1], b3[2]);

	printf("u1  %g %g %g\n", u1[0], u1[1], u1[2]);
	printf("u2  %g %g %g\n", u2[0], u2[1], u2[2]);
	printf("u3  %g %g %g\n", u3[0], u3[1], u3[2]);
*/

	for(int n=-nSites; n<=nSites; n++)
	{
		for (int m=-nSites; m<=nSites; m++)
		{
			for (int k=-nSites; k<=nSites; k++)
			{
				calcG_ab_nmk(alpha, beta, r, eta,
								n, m, k, partR, partQ);
				
				totalR += partR;
				totalQ += partQ;
			}
		}
	}

	return totalQ + totalR;
}

// Function that computes the contribution of the next term in the series
double DipoleEwald3D::calcG_ab_nextTerm(const int alpha, const int beta, const double* r, const double eta, const int nSites) 
{
	double totalQ = 0;
	double totalR = 0;
	
	double partQ, partR;

	// full k faces
	for(int n=-(nSites+1); n<=(nSites+1); n++)
	{
		for (int m=-(nSites+1); m<=(nSites+1); m++)
		{
				calcG_ab_nmk(alpha, beta, r, eta,
								n, m, -(nSites+1), partR, partQ);
				totalR += partR;
				totalQ += partQ;

				calcG_ab_nmk(alpha, beta, r, eta,
								n, m,   (nSites+1), partR, partQ);
				totalR += partR;
				totalQ += partQ;
		}
	}

	// m faces bounded by k
	for(int n=-(nSites+1); n<=(nSites+1); n++)
	{
		for (int k=-(nSites); k<=(nSites); k++)
		{
				calcG_ab_nmk(alpha, beta, r, eta,
								n, -(nSites+1), k, partR, partQ);
				totalR += partR;
				totalQ += partQ;

				calcG_ab_nmk(alpha, beta, r, eta,
								n,  (nSites+1), k, partR, partQ);
				totalR += partR;
				totalQ += partQ;
		}
	}

	// n faces bounded by k and m
	for(int m=-(nSites); m<=(nSites); m++)
	{
		for (int k=-(nSites); k<=(nSites); k++)
		{
				calcG_ab_nmk(alpha, beta, r, eta,
								-(nSites+1), m, k, partR, partQ);
				totalR += partR;
				totalQ += partQ;

				calcG_ab_nmk(alpha, beta, r, eta,
								 (nSites+1), m, k, partR, partQ);
				totalR += partR;
				totalQ += partQ;
		}
	}

	return totalQ + totalR; 
}



// support function
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

void DipoleEwald3D::push(lua_State* L)
{
	luaT_push<DipoleEwald3D>(L, this);
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

	r3 = lua_getNdouble(L, 3, b, 2+r1+r2, 0);
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

static int l_calctens(lua_State* L)
{
	LUA_PREAMBLE(DipoleEwald3D, e, 1);	

	const char* errmsg1 = "first argument of calculateTensorElement must be AB pair: `XX', `XY', ... `ZZ'";
	
	if(!lua_isstring(L, 2))
		return luaL_error(L, errmsg1);
	
	const char* AB = lua_tostring(L, 2);
	
	if(strlen(AB) < 2)
		return luaL_error(L, errmsg1);

	int ab[2];

	for(int i=0; i<2; i++)
	{
		switch(AB[i])
		{
			case 'x':
			case 'X':
				ab[i] = 0;
				break;
			case 'y':
			case 'Y':
				ab[i] = 1;
				break;
			case 'z':
			case 'Z':
				ab[i] = 2;
				break;
			default:
				return luaL_error(L, errmsg1);
	
		}
	}
	
	int r[3];
	int r1 = lua_getNint(L, 3, r, 3, 0);

	double tau = 0.001;
	if(lua_isnumber(L, 3+r1))
	{
		tau = lua_tonumber(L, 3+r1);
	}
	
	double elem;
	bool ok = e->calcTensorElement(ab[0], ab[1], r, elem, tau); 
	
	if(!ok)
	{
		return luaL_error(L, "Failed to calculate real/reciprocal partition point (eta) for series limits. Try a different tau value for a tolerance (current tau = %g)", tau);
	}

	lua_pushnumber(L, elem);
	return 1;
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
		lua_pushstring(L, "1 String, 1 *3Vector*, optionally 1 number: String is AB pair (`XX', `XY', ..., `ZZ') the *3Vector* represents offsets from zero. It takes 3 integer values each ranging from 0 to n-1 where n is the number of unit cells in the lattice in a given direction. The last optional number controls the calculation of an optimal eta value which partitions the series into real and reciprocal sums.");
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
		{"__tostring", l_tostring},
		{"setUnitCell",         l_setunitcell},
		{"unitCell", l_getunitcell},
		{"calculateTensorElement", l_calctens},
		{"latticeSize", l_getlatticesize},
		{"setLatticeSize", l_setlatticesize},
		
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
