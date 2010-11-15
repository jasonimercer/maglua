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

void lua_pushDipole(lua_State* L, Dipole* d);
Dipole* checkDipole(lua_State* L, int idx);
void registerDipole(lua_State* L);


#endif
