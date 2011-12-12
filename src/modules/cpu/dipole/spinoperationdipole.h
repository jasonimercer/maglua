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

#include "spinoperation.h"
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

class Dipole : public LongRange
{
public:
	Dipole(int nx=32, int ny=32, int nz=1);
	virtual ~Dipole();
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ);
};

DIPOLE_API void lua_pushDipole(lua_State* L, Encodable* d);
DIPOLE_API Dipole* checkDipole(lua_State* L, int idx);
DIPOLE_API void registerDipole(lua_State* L);


#endif
