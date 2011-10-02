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

#include "luacommon.h"
#include "encodable.h"
#include "jthread.h"

using namespace std;

class SpinSystem : public Encodable
{
public:
	SpinSystem(const int nx, const int ny, const int nz);
	~SpinSystem();

// 	SpinSystem* copy(lua_State* L);
// 	bool copyFrom(lua_State* L, SpinSystem* src);
// 	bool copySpinsFrom(lua_State* L, SpinSystem* src);
// 	bool copyFieldsFrom(lua_State* L, SpinSystem* src);
	
	void set(const int px, const int py, const int pz, const double x, const double y, const double z);
	void set(const int idx, double x, const double y, const double z);
	int  getidx(const int px, const int py, const int pz) const ;
	bool member(const int px, const int py, const int pz) const ;
	void sumFields();
	
	void zeroFields();
	void zeroField(int slot);
// 	bool addFields(double mult, SpinSystem* addThis);
	
	int getSlot(const char* name);
	static const char* slotName(int index);
	
	void getNetMag(double* v4);
	
// 	void diff(SpinSystem* other, double* v4);
	
	/* d_ = device (GPU) 
	 * h_ = host (CPU)
	 */
	double* d_x;
	double* d_y;
	double* d_z;
	double* d_ms; // spin length
	
	double* h_x;
	double* h_y;
	double* h_z;
	double* h_ms; // spin length
	
// 	double* h_x;
// 	double* h_y;
// 	double* h_z;


	// temporary workspaces on the device
	// these are used when kernels need to
	// split into multiple parts to
	// keep register counts down.
	double* d_ws1;
	double* d_ws2;
	double* d_ws3;
	double* d_ws4;
	
	double** d_hx;
	double** d_hy;
	double** d_hz;

	double** h_hx;
	double** h_hy;
	double** h_hz;

	bool* slot_used;
	int* extra_data; //used for site specific lua data
	
// 	int start_thread(int idx, void *(*start_routine)(void*), void* arg);
// 	JThread** jthreads;
	
	double alpha;
	double gamma;
	double dt;
	
	int nx, ny, nz;
	int refcount;

	int nxyz;
	int nslots;

	double time;
	
	void encode(buffer* b) const;
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

SpinSystem* checkSpinSystem(lua_State* L, int idx);
void lua_pushSpinSystem(lua_State* L, SpinSystem* ss);
void registerSpinSystem(lua_State* L);

#endif
