/********************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**********************************************************************/

#ifndef SPINOPERATIONLONGRANGE
#define SPINOPERATIONLONGRANGE

#include "spinoperation.h"
#include "array.h"

#ifdef WIN32
 #ifdef LONGRANGE_EXPORTS
  #define LONGRANGE_API __declspec(dllexport)
 #else
  #define LONGRANGE_API __declspec(dllimport)
 #endif
#else
 #define LONGRANGE_API 
#endif


using namespace std;

class LONGRANGE_API LongRange3D : public SpinOperation
{
public:
	LongRange3D(const char* name="LongRange3D", const int field_slot=DIPOLE_SLOT, int nx=1, int ny=1, int nz=1, const int encode_tag=hash32("LongRange3D"));
	virtual ~LongRange3D();
	
	LINEAGE2("LongRange3D", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);

	virtual const char* getSlotName() {return "LongRange3D";}

	bool apply(SpinSystem* ss);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

	double g; //scale

	virtual void init();
	virtual void deinit();
	
	dArray* getAB(const char* AB);
	void setAB(const char* AB, dArray* newArray);

	dArray* XX;
	dArray* XY;
	dArray* XZ;
	
	dArray* YX;
	dArray* YY;
	dArray* YZ;

	dArray* ZX;
	dArray* ZY;
	dArray* ZZ;

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
	dcArray* qXX;
	dcArray* qXY;
	dcArray* qXZ;

	dcArray* qYX;
	dcArray* qYY;
	dcArray* qYZ;
	
	dcArray* qZX;
	dcArray* qZY;
	dcArray* qZZ;

	dcArray* ws1;
	dcArray* ws2;
	dcArray* wsX;
	dcArray* wsY;
	dcArray* wsZ;

	int longrange_ref;
	int function_ref;
};


#endif
