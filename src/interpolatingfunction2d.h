#ifndef INTERPOLATINGFUCTION2D_H
#define INTERPOLATINGFUCTION2D_H

#include "luacommon.h"
#include <vector>
using namespace std;

class InterpolatingFunction2D
{
public:
	InterpolatingFunction2D();
	~InterpolatingFunction2D();

	void addData(const double inx, const double iny, const double out);
	bool getValue(double inx, double iny, double* out);
	bool compile();
	int refcount;

	double xmin,  ymin; 
	double xmax,  ymax; 

	bool compiled;

private:
	class triple
	{
	public:
		triple(double X=0, double Y=0, double Z=0) : x(X), y(Y), z(Z) {};
		double x, y, z;
	};

	void getixiy(double x, double y, int *v2);

	int getidx(double x, double y);


	double xstep, ystep;
	int nx, ny;

	double* data;

	vector < triple > rawdata;
};

InterpolatingFunction2D* checkInterpolatingFunction2D(lua_State* L, int idx);
void registerInterpolatingFunction2D(lua_State* L);
// 
#endif

