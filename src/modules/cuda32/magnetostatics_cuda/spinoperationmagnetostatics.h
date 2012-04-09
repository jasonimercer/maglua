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
#include "../longrange_cuda/spinoperationlongrange.h"

using namespace std;

class MagnetostaticCuda : public LongRangeCuda
{
public:
	MagnetostaticCuda(int nx=32, int ny=32, int nz=1);
	virtual ~MagnetostaticCuda();
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	double volumeDimensions[3];
	double crossover_tolerance; //calculations crossover from magnetostatics to dipole
	
	void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ);

	LINEAGE3("Magnetostatic", "LongRange", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);	
	
};


#endif
