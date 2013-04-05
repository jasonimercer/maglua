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

#ifndef SPINOPERATIONTHERMAL
#define SPINOPERATIONTHERMAL

#include "spinoperation.h"
#include "array.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef THERMAL_EXPORTS
  #define THERMAL_API __declspec(dllexport)
 #else
  #define THERMAL_API __declspec(dllimport)
 #endif
#else
 #define THERMAL_API 
#endif

class RNG;
class LLG;
class THERMAL_API Thermal : public SpinOperation
{
public:
	Thermal(int nx=32, int ny=32, int nz=1);
	virtual ~Thermal();
	
	LINEAGE2("Thermal", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual const char* getSlotName() {return "Thermal";}

	bool apply(SpinSystem* ss, RNG* rand);
	bool apply(SpinSystem* ss) {return apply(ss, 0);}

	void scaleSite(int px, int py, int pz, double strength);

	double temperature;
	dArray* scale;
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	RNG* myRNG;
};

#endif
