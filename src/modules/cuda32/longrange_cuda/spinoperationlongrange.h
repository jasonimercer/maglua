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

#ifndef SPINOPERATIONLONGRANGECUDA
#define SPINOPERATIONLONGRANGECUDA

#include "spinoperation.h"
#include "longrange_kernel.hpp"


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef LONGRANGECUDA_EXPORTS
  #define LONGRANGECUDA_API __declspec(dllexport)
 #else
  #define LONGRANGECUDA_API __declspec(dllimport)
 #endif
#else
 #define LONGRANGECUDA_API 
#endif

using namespace std;

class LONGRANGECUDA_API LongRangeCuda : public SpinOperation
{
public:
	LongRangeCuda(const char* name="LongRange", const int field_slot=DIPOLE_SLOT, int nx=1, int ny=1, int nz=1, const int encode_tag=hash32("LongRange"));
	virtual ~LongRangeCuda();
	
	LINEAGE2("LongRange", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);
	
	
	bool apply(SpinSystem* ss);
	bool applyToSum(SpinSystem* ss);
	
	double ABC[9];
	double g;
	int gmax;
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

	void init();
	void deinit();
	
	virtual void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ) {};

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
	void loadMatrix();
	bool getPlan();
	bool newHostData;
	bool matrixLoaded;
	
	JM_LONGRANGE_PLAN* plan;

	double* XX;
	double* XY;
	double* XZ;
	double* YY;
	double* YZ;
	double* ZZ;
};


#endif
