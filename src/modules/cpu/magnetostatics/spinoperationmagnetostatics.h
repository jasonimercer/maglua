/******************************************************************************
* Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#ifndef SPINOPERATIONMAGNETOSTATICS
#define SPINOPERATIONMAGNETOSTATICS

#include "spinoperation.h"
#include "../longrange/spinoperationlongrange.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef MAGNETOSTATICS_EXPORTS
  #define MAGNETOSTATICS_API __declspec(dllexport)
 #else
  #define MAGNETOSTATICS_API __declspec(dllimport)
 #endif
#else
 #define MAGNETOSTATICS_API 
#endif

using namespace std;

class Magnetostatic : public LongRange
{
public:
	Magnetostatic(int nx=32, int ny=32, int nz=1);
	virtual ~Magnetostatic();
	
	LINEAGE3("Magnetostatic", "LongRange", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual const char* getSlotName() {return "Magnetostatic";}


	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	double volumeDimensions[3];
	double crossover_tolerance; //calculations crossover from magnetostatics to dipole

	void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ);
};


#endif
