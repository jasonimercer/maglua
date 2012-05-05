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

#ifndef SPINSYSTEM
#define SPINSYSTEM

#include "maglua.h"
#include "luabaseobject.h"
#include "array.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
#endif

using namespace std;

class CORE_API SpinSystem : public LuaBaseObject
{
public:
	SpinSystem(const int nx=32, const int ny=32, const int nz=32);
	~SpinSystem();

	LINEAGE1("SpinSystem")
	static const luaL_Reg* luaMethods();
	virtual void push(lua_State* L);
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	SpinSystem* copy(lua_State* L);
	bool copyFrom(lua_State* L, SpinSystem* src);
	bool copySpinsFrom(lua_State* L, SpinSystem* src);
	bool copyFieldsFrom(lua_State* L, SpinSystem* src);
	bool copyFieldFrom(lua_State* L, SpinSystem* src, int slot);
	
	void set(const int px, const int py, const int pz, const double x, const double y, const double z);
	void set(const int idx, double x, const double y, const double z);
	int  getidx(const int px, const int py, const int pz) const ;
	bool member(const int px, const int py, const int pz) const ;
	void sumFields();
	
	void zeroFields();
	bool addFields(double mult, SpinSystem* addThis);
	
	int getSlot(const char* name);
	static const char* slotName(int index);
	
	
	void getNetMag(double* v4);
	
	void diff(SpinSystem* other, double* v4);

	dArray* x;
	dArray* y;
	dArray* z;
	
	dcArray* qx;
	dcArray* qy;
	dcArray* qz;

	dArray** hx;
	dArray** hy;
	dArray** hz;

	bool* slot_used;
	int* extra_data; //used for site specific lua data

	dArray* ms; // spin length
	
	double alpha;
	double gamma;
	double dt;
	
	int nx, ny, nz;

	int nxyz;
	int nslots;

	double time;
	double fft_timeC[3]; //last time for fft of component i (x,y,z)
	
	void fft();
	void fft(int component);
		
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

private:
	void init();
	void deinit();

	dcArray* rx;
	dcArray* ry;
	dcArray* rz;
};

// CORE_API SpinSystem* checkSpinSystem(lua_State* L, int idx);
// CORE_API SpinSystem* lua_toSpinSystem(lua_State* L, int idx);
// CORE_API int lua_isSpinSystem(lua_State* L, int idx);
// CORE_API void lua_pushSpinSystem(lua_State* L, LuaBaseObject* ss);
// CORE_API void registerSpinSystem(lua_State* L);

#endif
