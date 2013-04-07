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

#ifndef SPINOPERATION
#define SPINOPERATION

// #define SUM_SLOT          0
// #define EXCHANGE_SLOT     1
// #define APPLIEDFIELD_SLOT 2
// #define ANISOTROPY_SLOT   3
// #define THERMAL_SLOT      4
// #define DIPOLE_SLOT       5
// #define NSLOTS            6

//#include <omp.h>
#include "maglua.h"
#include <string>
#include "luabaseobject.h"
#include <vector>
using namespace std;
class SpinSystem;

class CORECUDA_API SpinOperation : public LuaBaseObject
{
public:
	//encodetype will be phased out in favour of dynamic_cast<T>()
	SpinOperation(std::string Name, int slot, int nx, int ny, int nz, int encodetype=0);
	virtual ~SpinOperation();
	
	LINEAGE1("SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	virtual bool apply(SpinSystem* ss);
	virtual bool apply(SpinSystem** sss, int n);
	int getSite(int x, int y, int z);

	bool member(int px, int py, int pz);
	int  getidx(int px, int py, int pz);

	void getSpinSystemsAtPosition(lua_State* L, int pos, vector<SpinSystem*>& sss);
	double** getVectorOfVectors(SpinSystem** sss, int n, const char* tag, const char data, const char component='Q', const int field=0);
	double*  getVectorOfValues(SpinSystem** sss, int n, const char* tag, const char data, const double scale=1.0);
		
	int nx, ny, nz;
	int nxyz;
	const std::string& name();

	std::string errormsg;
	
	void  idx2xyz(int idx, int& x, int& y, int& z) const ;

	double global_scale;
// 	virtual void encode(buffer* b) = 0;
// 	virtual int  decode(buffer* b) = 0;

protected:
	void markSlotUsed(SpinSystem* ss);
	
	std::string operationName;
	int slot;
};

CORECUDA_API int lua_getNint(lua_State* L, int N, int* vec, int pos, int def);
CORECUDA_API int lua_getNdouble(lua_State* L, int N, double* vec, int pos, double def);
CORECUDA_API int lua_getnewargs(lua_State* L, int* vec3, int pos);
#endif
