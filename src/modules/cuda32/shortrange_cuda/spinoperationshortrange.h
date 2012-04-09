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

#ifndef SPINOPERATIONSHORTRANGECUDA
#define SPINOPERATIONSHORTRANGECUDA

#include "spinoperation.h"
#include "shortrange_kernel.hpp"


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef SHORTRANGECUDA_EXPORTS
  #define SHORTRANGECUDA_API __declspec(dllexport)
 #else
  #define SHORTRANGECUDA_API __declspec(dllimport)
 #endif
#else
 #define SHORTRANGECUDA_API 
#endif

#include <vector>
using namespace std;

class shortrange_data
{
public:
	shortrange_data() {x=0;y=0;z=0;value=0;offset=0;}
	shortrange_data(int X, int Y, int Z, float V) {x=X;y=Y;z=Z;value=V;offset=0;}
	shortrange_data(const shortrange_data& s) : x(s.x), y(s.y), z(s.z), value(s.value), offset(s.offset) {}
	int x, y, z, offset;
	float value;
};



class ShortRangeCuda : public SpinOperation
{
public:
	ShortRangeCuda(int nx=32, int ny=32, int nz=1);
	virtual ~ShortRangeCuda();
	
	LINEAGE2("ShortRange", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);
	
	bool apply(SpinSystem* ss);
	bool applyToSum(SpinSystem* ss);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

	void init();
	void deinit();

	double getXX(int ox, int oy, int oz);
	void   setXX(int ox, int oy, int oz, double value);

	double getXY(int ox, int oy, int oz);
	void   setXY(int ox, int oy, int oz, double value);
	
	double getXZ(int ox, int oy, int oz);
	void   setXZ(int ox, int oy, int oz, double value);

	double getYY(int ox, int oy, int oz);
	void   setYY(int ox, int oy, int oz, double value);

	double getYZ(int ox, int oy, int oz);
	void   setYZ(int ox, int oy, int oz, double value);

	double getZZ(int ox, int oy, int oz);
	void   setZZ(int ox, int oy, int oz, double value);

	double getAB(int matrix, int ox, int oy, int oz);
	void   setAB(int matrix, int ox, int oy, int oz, double value);
private:
	void compile();
	bool getPlan();
	bool newHostData;
	
	vector<shortrange_data> AB[6];

	int* d_ABoffset[6];
	float* d_ABvalue[6];
	int* h_ABoffset[6];
	float* h_ABvalue[6];
	int  ABcount[6];
	
};


#endif
