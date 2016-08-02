/******************************************************************************
* Copyright (C) 2008-2012 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifndef SPINOPERATIONDIPOLE2D
#define SPINOPERATIONDIPOLE2D

#include "../longrange2d/spinoperationlongrange2d.h"
#include "array.h"

#ifdef WIN32
 #ifdef DIPOLE2D_EXPORTS
  #define DIPOLE2D_API __declspec(dllexport)
 #else
  #define DIPOLE2D_API __declspec(dllimport)
 #endif
#else
 #define DIPOLE2D_API 
#endif


// moving ABC and gmax etc to lua

using namespace std;

class DIPOLE2D_API Dipole2D : public LongRange2D
{
public:
	Dipole2D(int nx=1, int ny=1, int nz=1, const int encode_tag=hash32("LongRange2D"));
	virtual ~Dipole2D();
	
	LINEAGE3("Dipole2D", "LongRange2D", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
};


#endif
