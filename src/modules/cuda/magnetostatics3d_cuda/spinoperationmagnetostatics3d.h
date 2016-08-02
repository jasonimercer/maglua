/******************************************************************************
* Copyright (C) 3008-3011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifndef SPINOPERATIONMAGNETOSTATICS3D
#define SPINOPERATIONMAGNETOSTATICS3D

#include "../longrange3d_cuda/spinoperationlongrange3d.h"
#include "array.h"

#ifdef WIN32
 #ifdef MAGNETOSTATICS3D_EXPORTS
  #define MAGNETOSTATICS3D_API __declspec(dllexport)
 #else
  #define MAGNETOSTATICS3D_API __declspec(dllimport)
 #endif
#else
 #define MAGNETOSTATICS3D_API 
#endif


// moving ABC and gmax etc to lua

using namespace std;

class MAGNETOSTATICS3D_API Magnetostatics3D : public LongRange3D
{
public:
	Magnetostatics3D(int nx=1, int ny=1, int nz=1, const int encode_tag=hash32("Magnetostatics3D"));
	virtual ~Magnetostatics3D();
	
	LINEAGE3("Magnetostatics3D", "LongRange3D", "SpinOperation")
	double getGrainSize();
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual const char* getSlotName() {return "Magnetostatics3D";}
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
};

#endif
