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

#ifndef SPINOPERATIONDIPOLE
#define SPINOPERATIONDIPOLE

#include "../longrange/spinoperationlongrange.h"

#ifdef WIN32
 #ifdef DIPOLE_EXPORTS
  #define DIPOLE_API __declspec(dllexport)
 #else
  #define DIPOLE_API __declspec(dllimport)
 #endif
#else
 #define DIPOLE_API 
#endif


#include <complex>
#include <fftw3.h>

using namespace std;

class DIPOLE_API Dipole : public LongRange
{
public:
	Dipole(int nx=32, int ny=32, int nz=1);
	virtual ~Dipole();
	
	LINEAGE3("Dipole", "LongRange", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual const char* getSlotName() {return "Dipole";}


	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ);
};


#endif
