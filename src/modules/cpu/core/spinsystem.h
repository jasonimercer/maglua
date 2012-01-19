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

#include <complex>
#include <fftw3.h>
#include "luacommon.h"
#include "encodable.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
#endif

using namespace std;

class CORE_API SpinSystem : public Encodable
{
public:
	SpinSystem(const int nx=32, const int ny=32, const int nz=32);
	~SpinSystem();

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
	
	double* x;
	double* y;
	double* z;
	
	complex<double>* qx;
	complex<double>* qy;
	complex<double>* qz;

	double** hx;
	double** hy;
	double** hz;
	bool* slot_used;
	int* extra_data; //used for site specific lua data

	double* ms; // spin length
	
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
	double fft_time; //this is the time of the last fft
	
	void fft();
	
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

private:
	void init_fft();
	
	void init();
	void deinit();
	fftw_plan r2q;

	complex<double>* rx;
	complex<double>* ry;
	complex<double>* rz;
	
};

CORE_API SpinSystem* checkSpinSystem(lua_State* L, int idx);
CORE_API SpinSystem* lua_toSpinSystem(lua_State* L, int idx);
CORE_API int lua_isSpinSystem(lua_State* L, int idx);
CORE_API void lua_pushSpinSystem(lua_State* L, Encodable* ss);
CORE_API void registerSpinSystem(lua_State* L);

#endif
