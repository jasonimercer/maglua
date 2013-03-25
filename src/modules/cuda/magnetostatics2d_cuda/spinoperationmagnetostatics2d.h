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

#ifndef SPINOPERATIONMAGNETOSTATICS2D
#define SPINOPERATIONMAGNETOSTATICS2D

#include "../longrange2d_cuda/spinoperationlongrange2d.h"
#include "array.h"

#ifdef WIN32
 #ifdef MAGNETOSTATICS2D_EXPORTS
  #define MAGNETOSTATICS2D_API __declspec(dllexport)
 #else
  #define MAGNETOSTATICS2D_API __declspec(dllimport)
 #endif
#else
 #define MAGNETOSTATICS2D_API 
#endif


// moving ABC and gmax etc to lua

using namespace std;

class MAGNETOSTATICS2D_API Magnetostatics2D : public LongRange2D
{
public:
	Magnetostatics2D(const char* name="Magnetostatics2D", const int field_slot=DIPOLE_SLOT, int nx=1, int ny=1, int nz=1, const int encode_tag=hash32("LongRange2D"));
	virtual ~Magnetostatics2D();
	
	LINEAGE3("Magnetostatics2D", "LongRange2D", "SpinOperation")
	double getGrainSize(const int layer);
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
};


#endif
