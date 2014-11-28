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
#include "vec_cs.h"
#include <vector>

#include "../../cpu/core/spinsystem.h"
#include "array.h"


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

	LINEAGE1("MEP");
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L, int base=1);
	static int help(lua_State* L);

	int l_getCoordinateSystems(lua_State* L); // all available
	int l_setCoordinateSystem(lua_State* L, int idx); 
	int l_getCoordinateSystem(lua_State* L); // current cs

        int lua_getspinsystem(lua_State*);
        int lua_setspinsystem(lua_State*);
        int lua_setEnergyFunction(lua_State* L);

	void init();
	void deinit();

	int ref_data;
        int ref_energy_function;

        SpinSystem* ss;

	vector<VectorCS> state_path;
	// vector<double> state_xyz_path;
	
	vector<double> d12;  //this is the distance between hyperpoints
	double d12_total;
	bool good_distances; //dirty bit for good d12 vector
	double calculateD12(); // calcs pairwise distances, returns total distance. Will cache good results
	
	
	// move scales how much a point is allowed to move by. 
	// often the end points have a move of 0 
	vector<double> image_site_mobility; 
	vector<int> sites; //x,y,z idx
	
	vector<VectorCS> path_tangent;
	vector<VectorCS> force_vector;
	vector<double> energies;

	// using to look for curvatature changes = 0 which may be important sites.
	// hopefully we can resample at a finer resolution near these points
	// to better make use of images along the path. 
	int calculatePathEnergyNDeriv(lua_State* L, int n, vector<double>& nderiv);
	
	void setImageSiteMobility(const int image, const int site, double mobility);
	double getImageSiteMobility(const int image, const int site);

	double epsilon; // differentiation scale factor

	double beta; //step size
	bool energy_ok;
	
	void addSite(int x, int y, int z);
	void computeTangent(const int p1, const int p2, const int dest);

	double distanceBetweenHyperPoints(int p1, int p2);
	double distanceBetweenPoints(int p1, int p2, int site);
	
	void interpolateVectorCS(const VectorCS& v1, const VectorCS& v2, const double ratio, VectorCS& dest, bool backwards);
	void interpolateVectorCS(const vector<VectorCS>& v1, const vector<VectorCS>& v2, const double ratio, vector<VectorCS>& dest, bool backwards);


	void interpolatePoints(const int p1, const int p2, const int site, const double ratio, vector<VectorCS>& dest, const double jitter, bool backwards);
	void interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<VectorCS>& dest, const double jitter, bool* backwards);

	int calculateEnergyGradients(lua_State* L);

	void internal_copy_to(MEP* dest);

	int relaxSinglePoint_SteepestDecent(lua_State* L);
	int relaxSinglePoint_expensiveDecent(lua_State* L, int point, double h, int steps);
	int expensiveEnergyMinimization(lua_State* L, int point, double h, int steps);
	int expensiveGradientMinimization(lua_State* L, int point, double h, int steps);

//	int relaxSinglePoint(lua_State* L);
// 	int relaxSaddlePoint(lua_State* L);

	void projForcePerpSpins(lua_State* L); //project force onto subspace perpendicular to spin direction
	void projForcePerpPath(lua_State* L); //project force onto subspace perpendicular to path direction
	void projForcePath(lua_State* L); //project force onto vector parallel to the path direction
	
	int applyForces(lua_State* L);
	
	void randomize(const double magnitude);
	
	int calculateEnergies(lua_State* L);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	int resampleStatePath(lua_State* L);
// 	int resampleStateXYZPath(lua_State* L, vector<double>& points);
	
	int numberOfImages();
	int numberOfSites();
	int _resize(lua_State* L, int base);
	int _swap(lua_State* L, int base);
	int _copy(lua_State* L, int base);
	
	void setSiteSpin(int* site3, const double* mm);
	void setSiteSpin(int* site3, const VectorCS& mm);
	void setAllSpins(const vector<VectorCS>& m);
	void getAllSpins(vector<VectorCS>& m);

	double problemScale();
	double getEnergy(lua_State* L);
	double vecEnergy(lua_State* L, vector<VectorCS>& vec);

	// void getSiteSpin(lua_State* L, int get_index, int* site3, double* m3);
	void getSiteSpin(int* site3, vector<double>& v);
	void getSiteSpin(int* site3, VectorCS& m3);

	
	void saveConfiguration(vector<double>& buffer);
	void loadConfiguration(vector<double>& buffer);

	void getPoint(int p, vector<VectorCS>& dest);
	void setPoint(int p, vector<VectorCS>& src);

	void getForcePoint(int p, vector<VectorCS>& dest);
	void setForcePoint(int p, vector<VectorCS>& src);

	void getPathTangentPoint(int p, vector<VectorCS>& dest);
	void setPathTangentPoint(int p, vector<VectorCS>& src);

	int l_angle_between_pointsite(lua_State* L, int base);

	VectorCS getPointSite(int p, int s) {VectorCS c(0,0,0,Undefined); if(validPointSite(p,s)) getPointSite(p,s,c); return c;}
	void getPointSite(int p, int s, VectorCS& dest);
	void setPointSite(int p, int s, VectorCS  src);

	CoordinateSystem getCSAt(int p, int s);

	int validPointSite(int p, int s)
	{
		return (p >= 0 && s >= 0 && p < numberOfImages() && s < numberOfSites());
	}


	double absoluteDifference(MEP* other, int point, double& max_diff, int& max_idx);

	int maxpoints(lua_State* L);
	
	void computePointSecondDerivative(lua_State* L, int p, double* derivsAB);
	double computePointSecondDerivativeAB(lua_State* L, int p, int c1, int c2, double dc1=-1, double dc2=-1);

	void computePointFirstDerivative(lua_State* L, int p, const double p_energy, vector<VectorCS>& d);

	double computePointFirstDerivativeC(lua_State* L, int p, const double p_energy, int coord);

	void computeVecFirstDerivative(lua_State* L, vector<VectorCS>& vec, const double vec_energy, vector<VectorCS>& dest);

	double computeVecFirstDerivativeC(lua_State* L, vector<VectorCS>& vec, const double vec_energy, int coord);

	int uniqueSites(lua_State* L);
	int slidePoint(lua_State* L);
	int classifyPoint(lua_State* L);


	int l_classifyPoint(lua_State* L, int base);

	// get energies about a point
	void pointNeighbourhood(const int point, lua_State* L, const vector<double>& h, vector<double>& e);

	// get energies about a state
	void stateNeighbourhood(const vector<VectorCS>& state, lua_State* L, const vector<double>& h, vector<double>& e);

	// type: -1=min, 0=saddle, 1=max
	double classifyNeighbourhood(int type, const vector<double>& eee_h);

        void rotatePathAboutBy(const VectorCS& n, const double rad);
        int  l_rotatePathAboutBy(lua_State* L, int base);



	int relax_direction_fail_max;

	int anglesBetweenPoints(lua_State* L);

	bool fixedRadius;

	bool equalByArcLength(int a, int b, double allowable);
	bool equalByAngle(int a, int b, double radians);
private:
	void make_path_size(const int n);

	void computePointGradAtSite(lua_State* L, int p, int s, double* grad3);
	// void writePathPoint(lua_State* L, int set_index, double* vxyz);
	void listDerivN(vector<double>& dest, vector<double>& src);

	double rightDeriv(vector<double>& src, int i);
	double leftDeriv(vector<double>& src, int i);
	
	void printState();

        void get_site_ss(int px, int py, int pz, double* v3);
        bool set_site_ss(int px, int py, int pz, double* v3);
        double get_energy_ss(lua_State* L);

};

#endif
