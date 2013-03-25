#include "luabaseobject.h"

#ifdef WIN32
#define strcasecmp(A,B) _stricmp(A,B)
#define strncasecmp(A,B,C) _strnicmp(A,B,C)
#pragma warning(disable: 4251)

#ifdef EWALD3D_EXPORTS
#define EWALD3D_API __declspec(dllexport)
#else
#define EWALD3D_API __declspec(dllimport)
#endif
#else
#define EWALD3D_API
#endif

#ifndef DIPOLEEWALD3D
#define DIPOLEEWALD3D

class EWALD3D_API DipoleEwald3D : public LuaBaseObject
{
public:
	DipoleEwald3D()
		: LuaBaseObject(hash32(lineage(0))),
		  nSites(10), tau(1e-3), d1(8), d2(8), d3(8), eta(0.5) //sane default values
	{
		//cubic unit cell
		a1[0] = 1; a1[1] = 0; a1[2] = 0;
		a2[0] = 0; a2[1] = 1; a2[2] = 0;
		a3[0] = 0; a3[1] = 0; a3[2] = 1;
		calcSupportVectors();
	}
	
	LINEAGE1("DipoleEwald3D")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);

	void encode(buffer* b);
	int  decode(buffer* b);


	// a,b: X=0, Y=1, Z=2
	bool calcTensorElement(const int alpha, const int beta, const int* r3, double& element, const int nSites=10, const double tau = 0.001);
	bool calcTensorElementRealR(const int alpha, const int beta, const double* r, double& element, const int nSites=10,  const double tau = 0.001);

	void setUnitCell(const double* A1, const double* A2, const double* A3);
	
	void setLatticeSiteCount(const int nx, const int ny, const int nz);

	void calcSupportVectors();
	
	//system size (lattice sites in X, Y and Z directions)
	int d1, d2, d3;

	double a1[3], a2[3], a3[3]; //real space vectors (unit cell basis vectors)

	double getEta() const {return eta;}

	void setTau(const double t) {if(t > 0) {tau = t; gotGoodEta = false;}}
	double getTau() const {return tau;}
	
	void setNSites(int n) {if(n>1){nSites = n; gotGoodEta = false;}}
	int getNSites() const {return nSites;}
	
	double getVolume() const {return vu;}


	bool gotGoodEta; //when parameters change we need a new eta
private:

	double b1[3], b2[3], b3[3]; //reciprocal lattice vectors
	
	//real space super-lattice space vectors
	double u1[3], u2[3], u3[3]; 
	
	//Volume of 3D cell formed by {u1,u2,u3}
	double vu;

	int nSites; //Number of terms to sum over in each direction {a1/b1, a2/b2, a3/b3}

	double tau; // tolerance for eta search

	double eta; // real/Q split

	
	//Difference in Interaction Matrices by summing over nSites and (nSites+1)
	double delGXXleft, delGXXright, delGXXmid, delGXXx2, delGXXx4; 
	
	//Function that calculates individual terms of GAB
	double calcG_ab (int alpha, int beta, const double* r, const double eta, const int nSites); 
	
	// fast method for calcG_ab(..nSites+1) - calcG_ab(...nSites)
	// by only calculating the surface terms
	double calcG_ab_nextTerm(const int alpha, const int beta, const double* r, const double eta, const int nSites);

	//Function that calculates parts of individual terms of GAB
	void calcG_ab_nmk(	const int alpha, const int beta,
					  const double* r, const double eta,
					  const int n, const int m, const int k,  //this is the part of the term requested
					  double& real_part, double& recip_part);

	//Function that calculates parts of individual terms of GAB - special case: r=0
	void calcG_ab_nmk0(	const int alpha, const int beta,
						const double eta,
						const int n, const int m, const int k,  //this is the part of the term requested
						double& real_part, double& recip_part);

	// search for a good split between real/recip series
	bool  findEta(const double* r3, const int nSites, double tau = 0.001);
	
	// recursive version of splitting with caching of evaluations
	bool _findEta(const int n, const double* r3,
				  const double a, const double fa,
				  const double b, const double fb,
				  const double c, const double fc,
				  const int nSites, const double tau);
};

#endif
