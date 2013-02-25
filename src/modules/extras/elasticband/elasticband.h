#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ELASTICBAND_EXPORTS
  #define ELASTICBAND_API __declspec(dllexport)
 #else
  #define ELASTICBAND_API __declspec(dllimport)
 #endif
#else
 #define ELASTICBAND_API 
#endif


#ifndef ELASTICBAND_DEF
#define ELASTICBAND_DEF

#include "luabaseobject.h"
#include <vector>

class ELASTICBAND_API ElasticBand : public LuaBaseObject
{
public:
	ElasticBand();
	~ElasticBand();

	LINEAGE1("ElasticBand")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L, int base=1);
	static int help(lua_State* L);

	void init();
	void deinit();

	int ref_data;

	int initializeEndpoints(lua_State* L);
	
	vector<double> state_xyz_path;
	vector<int> sites; //x,y,z idx
	

	vector<double> path_tangent;
	vector<double> force_vector;
	vector<double> energies;
	
	void addSite(int x, int y, int z);
	void computeTangent(const int p1, const int p2, const int dest);
	void calcSpringForces(double k);
	void calculateOffsetVector(double* vec, const int p1, const int p2);

	double distanceBetweenHyperPoints(int p1, int p2);
	double distanceBetweenPoints(int p1, int p2, int site);
	
	void interpolatePoints(const int p1, const int p2, const int site, const double ratio, vector<double>& dest, const double noise);
	void interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<double>& dest, const double noise);

	int calculateEnergyGradients(lua_State* L);

	void projForcePerpSpins(); //project force onto subspace perpendicular to spin direction
	void projForcePerpPath(); //project force onto subspace perpendicular to path direction
	void projForcePath(); //project force onto vector parallel to the path direction
	
	int calculateEnergies(lua_State* L);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	int resampleStateXYZPath(lua_State* L, int new_num_points, const double noise);
private:
	void make_path_size(const int n);

};

#endif
