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

using namespace std;

typedef struct work_space_device_memory
{
	int refcount;
	void* d_memory[5];
	size_t size[5];
} work_space_device_memory;

static work_space_device_memory WS_MEM = {0};
CORECUDA_API void  registerWS();
CORECUDA_API void  unregisterWS();
CORECUDA_API void  getWSMem(void** ptr1,   size_t size1, 
			   void** ptr2=0, size_t size2=0, 
			   void** ptr3=0, size_t size3=0,
			   void** ptr4=0, size_t size4=0,
			   void** ptr5=0, size_t size5=0);
CORECUDA_API void  getWSMem(double** ptr1,   size_t size1, 
			   double** ptr2=0, size_t size2=0, 
			   double** ptr3=0, size_t size3=0,
			   double** ptr4=0, size_t size4=0,
			   double** ptr5=0, size_t size5=0);


class CORECUDA_API SpinSystem : public LuaBaseObject
{
public:
	SpinSystem(const int nx=32, const int ny=32, const int nz=1);
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
	void zeroField(int slot);
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
};

#endif
