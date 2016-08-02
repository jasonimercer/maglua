/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifndef INTERPOLATINGFUCTION_H
#define INTERPOLATINGFUCTION_H

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}
#include <vector>
#include "luabaseobject.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef INTERPOLATE_EXPORTS
  #define INTERPOLATE_API __declspec(dllexport)
 #else
  #define INTERPOLATE_API __declspec(dllimport)
 #endif
#else
 #define INTERPOLATE_API 
#endif

using namespace std;

class InterpolatingFunction : public LuaBaseObject
{
public:
	InterpolatingFunction();
	~InterpolatingFunction();

	LINEAGE1("Interpolate")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	void addData(const double in, const double out);
	bool getValue(double in, double* out);
	bool getValue(double in, double& out);

	void encode(buffer* b);
	int  decode(buffer* b);
	
	double maxX();
	double minX();
	
	InterpolatingFunction* inverted();

        void clear();

//private:
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

// 
#endif

