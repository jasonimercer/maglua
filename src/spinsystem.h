#ifndef SPINSYSTEM
#define SPINSYSTEM

#include <complex>
#include <fftw3.h>
#include "luacommon.h"
#include "encodable.h"
using namespace std;

class SpinSystem : public Encodable
{
public:
	SpinSystem(int nx, int ny, int nz);
	~SpinSystem();
	
	void set(int px, int py, int pz, double x, double y, double z);
	void set(int idx, double x, double y, double z);
	int  getidx(int px, int py, int pz);
	bool member(int px, int py, int pz);
	void sumFields();
	
	void zeroFields();

	int getSlot(const char* name);

	void getNetMag(double* v4);
	
	double* x;
	double* y;
	double* z;
	
	complex<double>* qx;
	complex<double>* qy;
	complex<double>* qz;

	double** hx;
	double** hy;
	double** hz;
	
	double* ms; // spin length
	
	int nx, ny, nz;
	int refcount;

	int nxyz;
	int nslots;

	double time;
	
	void fft();
	
	
	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);

private:
	void init();
	void deinit();
	fftw_plan r2q;

	complex<double>* rx;
	complex<double>* ry;
	complex<double>* rz;

};

SpinSystem* checkSpinSystem(lua_State* L, int idx);
void lua_pushSpinSystem(lua_State* L, SpinSystem* ss);
void registerSpinSystem(lua_State* L);


#endif
