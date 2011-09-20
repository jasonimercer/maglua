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

#ifndef SPINOPERATIONDIPOLE
#define SPINOPERATIONDIPOLE

#include "spinoperation.h"

#include <cufft.h>

using namespace std;

class DipoleCuda : public SpinOperation
{
public:
	DipoleCuda(int nx, int ny, int nz);
	virtual ~DipoleCuda();
	
	bool apply(SpinSystem* ss);
	void getMatrices();
	
	void threadApply(SpinSystem* ss);
	
	double g;

	double ABC[9];
	int gmax;
	
	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);

private:
	void ifftAppliedForce(SpinSystem* ss);
	void collectIForces(SpinSystem* ss);

	bool hasMatrices;

	cufftDoubleComplex* srx;
	cufftDoubleComplex* sry;
	cufftDoubleComplex* srz;

	cufftDoubleComplex* hqx;
	cufftDoubleComplex* hqy;
	cufftDoubleComplex* hqz;

	cufftDoubleComplex* hrx;
	cufftDoubleComplex* hry;
	cufftDoubleComplex* hrz;


	cufftDoubleComplex* qXX;
	cufftDoubleComplex* qXY;
	cufftDoubleComplex* qXZ;

	cufftDoubleComplex* qYY;
	cufftDoubleComplex* qYZ;
	cufftDoubleComplex* qZZ;

	cufftHandle plan;
};

void lua_pushDipoleCuda(lua_State* L, DipoleCuda* d);
DipoleCuda* checkDipoleCuda(lua_State* L, int idx);
void registerDipoleCuda(lua_State* L);


#endif
