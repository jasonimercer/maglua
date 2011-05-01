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

#ifndef SPINOPERATION
#define SPINOPERATION

#define SUM_SLOT          0
#define EXCHANGE_SLOT     1
#define APPLIEDFIELD_SLOT 2
#define ANISOTROPY_SLOT   3
#define THERMAL_SLOT      4
#define DIPOLE_SLOT       5
#define NSLOTS            6

//#include <omp.h>
#include "luacommon.h"
#include <string>
#include "encodable.h"
class SpinSystem;

class SpinOperation : public Encodable
{
public:
	//encodetype will be phased out in favour of dynamic_cast<T>()
	SpinOperation(std::string Name, int slot, int nx, int ny, int nz, int encodetype=0);
	virtual ~SpinOperation();
	
	virtual bool apply(SpinSystem* ss) = 0;
	int getSite(int x, int y, int z);

	bool member(int px, int py, int pz);
	int  getidx(int px, int py, int pz);

	int nx, ny, nz;
	int nxyz;
	const std::string& name();
	int refcount;

	std::string errormsg;
	
	virtual void encode(buffer* b) const = 0;
	virtual int  decode(buffer* b) = 0;

protected:
	void markSlotUsed(SpinSystem* ss);
	
	std::string operationName;
	int slot;
};

int lua_getNint(lua_State* L, int N, int* vec, int pos, int def);
int lua_getNdouble(lua_State* L, int N, double* vec, int pos, double def);
int lua_getnewargs(lua_State* L, int* vec3, int pos);
#endif
