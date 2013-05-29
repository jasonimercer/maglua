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

// Custom implementation of the elastic band minimum energy pathway algorithm
//
// Based on
// Journal of Magnetism and Magnetic Materials 250 (2002) L12–L19
// and discussions with JvE.

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
	
	void interpolatePoints(const int p1, const int p2, const int site, const double ratio, vector<double>& dest, const double noise, const double* noise_vec3);
	void interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<double>& dest, const double noise, const double* noise_vec3);

	int calculateEnergyGradients(lua_State* L);

	int relaxSinglePoint(lua_State* L);

	void projForcePerpSpins(); //project force onto subspace perpendicular to spin direction
	void projForcePerpPath(); //project force onto subspace perpendicular to path direction
	void projForcePath(); //project force onto vector parallel to the path direction
	
	int applyForces(lua_State* L, double dt);
	
	void computePointSecondDerivative(lua_State* L, int p, double h, int set_index, int get_index, int energy_index, double* derivsAB);
	
	int calculateEnergies(lua_State* L);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	int resampleStateXYZPath(lua_State* L, int new_num_points, const double noise);
	
	int numberOfPoints();
	int numberOfSites();
	
	void setSiteSpin(lua_State* L, int set_index, int* site3, double* m3);
	void setSiteSpin(lua_State* L, int set_index, int* site3, double sx, double sy, double sz);
	void setAllSpins(lua_State* L, int set_index, double* m);
	void getAllSpins(lua_State* L, int get_index, double* m);

	
	double getEnergy(lua_State* L, int energy_index);
	
	void getSiteSpin(lua_State* L, int get_index, int* site3, double* m3);
	void getSiteSpin(lua_State* L, int get_index, int* site3, vector<double>& v);
	
	void saveConfiguration(lua_State* L, int get_index, vector<double>& buffer);
	void loadConfiguration(lua_State* L, int set_index, vector<double>& buffer);

	int maxpoints(lua_State* L);
	
private:
	void make_path_size(const int n);

	void computePointGradAtSite(lua_State* L, int p, int s, double epsilon, int set_index, int energy_index, double* grad3);
	void writePathPoint(lua_State* L, int set_index, double* vxyz);
};

#endif
