#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef _MEP_EXPORTS
  #define _MEP_API __declspec(dllexport)
 #else
  #define _MEP_API __declspec(dllimport)
 #endif
#else
 #define _MEP_API 
#endif


#ifndef _MEP_DEF
#define _MEP_DEF

#include "luabaseobject.h"
#include <vector>

// Custom implementation of the elastic band minimum energy pathway algorithm
//
// Based on
// Journal of Magnetism and Magnetic Materials 250 (2002) L12–L19
// and discussions with JvE.
// 
// The "elastic" part has been removed and replaced with a resampling method
// this makes this code more similar to the "string" methods

class _MEP_API MEP : public LuaBaseObject
{
public:
	MEP();
	~MEP();

	LINEAGE1("MEP")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L, int base=1);
	static int help(lua_State* L);

	void init();
	void deinit();

	int ref_data;

	vector<double> state_xyz_path;
	// move scales how much a point is allowed to move by. 
	// often the end points have a move of 0 
	vector<double> image_site_mobility; 
	vector<int> sites; //x,y,z idx
	
	vector<double> path_tangent;
	vector<double> force_vector;
	vector<double> energies;
	
	void setImageSiteMobility(const int image, const int site, double mobility);
	double getImageSiteMobility(const int image, const int site);

	double beta; //step size
	bool energy_ok;
	
	void addSite(int x, int y, int z);
	void computeTangent(const int p1, const int p2, const int dest);
// 	void computeTangent(lua_State* L, int get_index, int set_index, int energy_index, const int p);
	void calculateOffsetVector(double* vec, const int p1, const int p2);

	double distanceBetweenHyperPoints(int p1, int p2);
	double distanceBetweenPoints(int p1, int p2, int site);
	
	void interpolatePoints(const int p1, const int p2, const int site, const double ratio, vector<double>& dest, const double jitter);
	void interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<double>& dest, const double jitter);

	int calculateEnergyGradients(lua_State* L, int get_index, int set_index, int energy_index);

	void internal_copy_to(MEP* dest);

	int relaxSinglePoint(lua_State* L);
// 	int relaxSaddlePoint(lua_State* L);

	void projForcePerpSpins(lua_State* L, int get_index, int set_index, int energy_index); //project force onto subspace perpendicular to spin direction
	void projForcePerpPath(lua_State* L, int get_index, int set_index, int energy_index); //project force onto subspace perpendicular to path direction
	void projForcePath(lua_State* L, int get_index, int set_index, int energy_index); //project force onto vector parallel to the path direction
	
	int applyForces(lua_State* L);
	
	void randomize(const double magnitude);
	
	int calculateEnergies(lua_State* L, int get_index, int set_index, int energy_index);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	int resampleStateXYZPath(lua_State* L);
	
	int numberOfImages();
	int numberOfSites();
	
	void setSiteSpin(lua_State* L, int set_index, int* site3, double* m3);
	void setSiteSpin(lua_State* L, int set_index, int* site3, double sx, double sy, double sz);
	void setAllSpins(lua_State* L, int set_index, double* m);
	void getAllSpins(lua_State* L, int get_index, double* m);

	double problemScale();
	double getEnergy(lua_State* L, int energy_index);
	
	void getSiteSpin(lua_State* L, int get_index, int* site3, double* m3);
	void getSiteSpin(lua_State* L, int get_index, int* site3, vector<double>& v);
	
	void saveConfiguration(lua_State* L, int get_index, vector<double>& buffer);
	void loadConfiguration(lua_State* L, int set_index, vector<double>& buffer);

	double absoluteDifference(MEP* other, int point, double& max_diff);

	int maxpoints(lua_State* L);
	
	void computePointSecondDerivative(lua_State* L, int p, int set_index, int get_index, int energy_index, double* derivsAB);
	double computePointSecondDerivativeAB(lua_State* L, int p, int set_index, int get_index, int energy_index, int c1, int c2);
private:
	void make_path_size(const int n);

	void computePointGradAtSite(lua_State* L, int p, int s, int set_index, int energy_index, double* grad3);
	void writePathPoint(lua_State* L, int set_index, double* vxyz);
	
	void printState();
};

#endif
