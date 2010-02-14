#ifndef SPINOPERATIONDIPOLE
#define SPINOPERATIONDIPOLE

#include "spinoperation.h"

#include <complex>
#include <fftw3.h>

using namespace std;

class Dipole : public SpinOperation
{
public:
	Dipole(int nx, int ny, int nz);
	virtual ~Dipole();
	
	bool apply(SpinSystem* ss);
	void getMatrices();
	
	double g;

	double ABC[9];
	int gmax;
	
	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);

	private:
	void ifftAppliedForce(SpinSystem* ss);
	void collectIForces(SpinSystem* ss);

	bool hasMatrices;

	complex<double>* srx;
	complex<double>* sry;
	complex<double>* srz;

	complex<double>* hqx;
	complex<double>* hqy;
	complex<double>* hqz;

	complex<double>* hrx;
	complex<double>* hry;
	complex<double>* hrz;


	complex<double>* qXX;
	complex<double>* qXY;
	complex<double>* qXZ;

	complex<double>* qYY;
	complex<double>* qYZ;
	complex<double>* qZZ;


	fftw_plan forward;
	fftw_plan backward;
};

Dipole* checkDipole(lua_State* L, int idx);
void registerDipole(lua_State* L);


#endif
