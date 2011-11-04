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

#ifndef SPINOPERATIONMAGNETOSTATICSCUDA
#define SPINOPERATIONMAGNETOSTATICSCUDA

#include "spinoperation.h"
#include "kernel.hpp"


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef DIPOLECUDA_EXPORTS
  #define DIPOLECUDA_API __declspec(dllexport)
 #else
  #define DIPOLECUDA_API __declspec(dllimport)
 #endif
#else
 #define DIPOLECUDA_API 
#endif

using namespace std;

class MagnetostaticCuda : public SpinOperation
{
public:
	MagnetostaticCuda(int nx=32, int ny=32, int nz=1);
	virtual ~MagnetostaticCuda();
	
	bool apply(SpinSystem* ss);
	void getMatrices();
	
	double g;
	
	double ABC[9];
	int gmax;
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	double volumeDimensions[3];
	double crossover_tolerance; //calculations crossover from magnetostatics to dipole

private:
	void init();
	void deinit();

	void getPlan();

	JM_LONGRANGE_PLAN* plan;
};

void lua_pushMagnetostaticCuda(lua_State* L, Encodable* d);
MagnetostaticCuda* checkMagnetostaticCuda(lua_State* L, int idx);
void registerMagnetostaticCuda(lua_State* L);


#endif
