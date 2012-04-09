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
			   void** ptr2, size_t size2, 
			   void** ptr3, size_t size3,
			   void** ptr4, size_t size4,
			   void** ptr5, size_t size5);
CORECUDA_API void  getWSMem(double** ptr1,   size_t size1, 
			   double** ptr2=0, size_t size2=0, 
			   double** ptr3=0, size_t size3=0,
			   double** ptr4=0, size_t size4=0,
			   double** ptr5=0, size_t size5=0);
CORECUDA_API void  getWSMem(float** ptr1,   size_t size1, 
			   float** ptr2=0, size_t size2=0, 
			   float** ptr3=0, size_t size3=0,
			   float** ptr4=0, size_t size4=0,
			   float** ptr5=0, size_t size5=0);


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
	
	void getNetMag(double* v8);
	
	void diff(SpinSystem* other, double* v4);
	void ensureSlotExists(int slot);
	/* d_ = device (GPU) 
	 * h_ = host (CPU)
	 */
	float* d_x;
	float* d_y;
	float* d_z;
	float* d_ms; // spin length
	
	float* h_x;
	float* h_y;
	float* h_z;
	float* h_ms; // spin length

	
	float* h_ws1;
	
	float** d_hx;
	float** d_hy;
	float** d_hz;

	float** h_hx;
	float** h_hy;
	float** h_hz;

	bool* slot_used;
	int* extra_data; //used for site specific lua data
	
	double gamma;
	double alpha;
	double dt;
	
	int nx, ny, nz;

	int nxyz;

	double time;
	
	void encode(buffer* b);
	int  decode(buffer* b);

	lua_State* L;

	void sync_spins_dh(bool force=false);
	void sync_spins_hd(bool force=false);

	void sync_fields_dh(int field,bool force=false);
	void sync_fields_hd(int field,bool force=false);

	bool  new_host_spins;   // if host data is most recent
	bool* new_host_fields; 

	bool  new_device_spins;
	bool* new_device_fields;
	
private:
	void init();
	void deinit();
};

#endif
