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

#ifdef CUDA_VERSION
#define EXPORT_API CORECUDA_API
#else
#define EXPORT_API CORE_API
#endif

class EXPORT_API SpinSystem : public LuaBaseObject
{
public:
	SpinSystem(const int nx=32, const int ny=32, const int nz=32);
	~SpinSystem();

	LINEAGE1("SpinSystem")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	SpinSystem* copy(lua_State* L);
	bool copyFrom(lua_State* L, SpinSystem* src);
	bool copySpinsFrom(lua_State* L, SpinSystem* src);
	bool copyFieldsFrom(lua_State* L, SpinSystem* src);
	bool copyFieldFrom(lua_State* L, SpinSystem* src, const char* slot_name);
	void moveToward(SpinSystem* other, double r);

	void rotateToward(SpinSystem* other, double max_angle, dArray* max_by_site);
	
	void setSiteAlpha(const int px, const int py, const int pz, const double a);
	void setSiteAlpha(const int idx, double a);
	void setAlpha(const double a);

	void setSiteGamma(const int px, const int py, const int pz, const double g);
	void setSiteGamma(const int idx, double g);
	void setGamma(const double g);

	void set(const int px, const int py, const int pz, const double x, const double y, const double z);
	void set(const int idx, double x, const double y, const double z);
	int  getidx(const int px, const int py, const int pz) const ;
	void  idx2xyz(int idx, int& x, int& y, int& z) const ;
	bool member(const int px, const int py, const int pz) const ;
	void sumFields();
	
	void zeroFields();
	bool addFields(double mult, SpinSystem* addThis);
	

	bool sameSize(const SpinSystem* other) const;
	
	void getNetMag(dArray* site_scale1, dArray* site_scale2, dArray* site_scale3, double* v8, const double scale);
	
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

	char** registered_slot_names;

	int nslots;

	int register_slot_name(const char* name);
	int getSlot(const char* name);
	const char* slotName(int index);
	void ensureSlotExists(int slot);	
	
	dArray* getFeildArray(int component, const char* name);
	bool setFeildArray(int component, const char* name, dArray* a);
	
	int* extra_data_size; //used for site specific lua data
	char** extra_data;

	dArray* ms; // spin length
	
	double alpha;
	double gamma;
	double dt;
	
	dArray* site_alpha;
	dArray* site_gamma;
	
	int nx, ny, nz;

	int nxyz;

	double time;
	double fft_timeC[3]; //last time for fft of component i (x,y,z)
	
	void fft();
	void fft(int component);
	void invalidateFourierData();
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

private:
	void init();
	void deinit();

	dcArray* ws;
	dcArray* ws2;
	dArray* wsReal;
};

#endif
