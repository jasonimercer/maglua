#ifndef INTERPOLATINGFUCTION_H
#define INTERPOLATINGFUCTION_H

#include "luacommon.h"
#include <vector>
using namespace std;

class InterpolatingFunction
{
public:
	InterpolatingFunction();
	~InterpolatingFunction();

	void addData(const double in, const double out);
	bool getValue(double in, double* out);
	int refcount;	
private:
	class _node
	{
	public:
		_node(double x1, double y1, double x2, double y2);
		_node(_node* c0, _node* c1);
		~_node();

		bool inrange(const double test);

		_node* c[2];
		double x[2];
		double y[2];
		double m, cut;
	};
	
	void compile();

	bool compiled;

	vector <pair<double,double> > rawdata;
	_node* root;
};

InterpolatingFunction* checkInterpolatingFunction(lua_State* L, int idx);
void registerInterpolatingFunction(lua_State* L);
// 
#endif

