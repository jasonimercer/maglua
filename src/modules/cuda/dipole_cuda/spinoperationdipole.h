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

#ifndef SPINOPERATIONDIPOLECUDA
#define SPINOPERATIONDIPOLECUDA

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

class DipoleCuda : public SpinOperation
{
public:
	DipoleCuda(int nx, int ny, int nz);
	virtual ~DipoleCuda();
	
	bool apply(SpinSystem* ss);
	
	double g;

	double ABC[9];
	int gmax;
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

private:
	void getPlan();

	JM_LONGRANGE_PLAN* plan;
};

DIPOLECUDA_API void lua_pushDipoleCuda(lua_State* L, DipoleCuda* d);
DIPOLECUDA_API DipoleCuda* checkDipoleCuda(lua_State* L, int idx);
DIPOLECUDA_API void registerDipoleCuda(lua_State* L);


#endif
