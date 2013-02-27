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

#ifndef SPINOPERATIONLONGRANGE2D
#define SPINOPERATIONLONGRANGE2D

#include "spinoperation.h"
#include "array.h"

#ifdef WIN32
 #ifdef LONGRANGE2D_EXPORTS
  #define LONGRANGE2D_API __declspec(dllexport)
 #else
  #define LONGRANGE2D_API __declspec(dllimport)
 #endif
#else
 #define LONGRANGE2D_API 
#endif


// moving ABC and gmax etc to lua

using namespace std;

class LONGRANGE2D_API LongRange2D : public SpinOperation
{
public:
	LongRange2D(const char* name="LongRange2D", const int field_slot=DIPOLE_SLOT, int nx=1, int ny=1, int nz=1, const int encode_tag=hash32("LongRange2D"));
	virtual ~LongRange2D();
	
	LINEAGE2("LongRange2D", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);

	bool apply(SpinSystem* ss);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

	double* g; //scale per layer

	virtual void init();
	virtual void deinit();
	
	dArray* getLAB(int layer_dest, int layer_src, const char* AB);
	void setLAB(int layer_dest, int layer_src, const char* AB, dArray* newArray);

	// these are [destination layer][source layer]
	dArray*** XX;
	dArray*** XY;
	dArray*** XZ;
	
	dArray*** YX;
	dArray*** YY;
	dArray*** YZ;

	dArray*** ZX;
	dArray*** ZY;
	dArray*** ZZ;

	bool compileRequired;
	void compile();
	
	bool newDataRequired;
	void makeNewData();


	dcArray* srx;
	dcArray* sry;
	dcArray* srz;

	dcArray* hqx;
	dcArray* hqy;
	dcArray* hqz;

	dcArray* hrx;
	dcArray* hry;
	dcArray* hrz;

	// these are [destination layer][source layer]
	dcArray*** qXX;
	dcArray*** qXY;
	dcArray*** qXZ;

	dcArray*** qYX;
	dcArray*** qYY;
	dcArray*** qYZ;
	
	dcArray*** qZX;
	dcArray*** qZY;
	dcArray*** qZZ;

	dcArray* ws1;
	dcArray* ws2;

	int longrange_ref;
	int function_ref;
};


#endif
