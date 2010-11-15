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
	SpinSystem(const int nx, const int ny, const int nz);
	~SpinSystem();
	
	void set(const int px, const int py, const int pz, const double x, const double y, const double z);
	void set(const int idx, double x, const double y, const double z);
	int  getidx(const int px, const int py, const int pz) const ;
	bool member(const int px, const int py, const int pz) const ;
	void sumFields();
	
	void zeroFields();
	bool addFields(double mult, SpinSystem* addThis);
	bool copy(SpinSystem* src);
	
	int getSlot(const char* name);
	static const char* slotName(int index);
	
	
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
