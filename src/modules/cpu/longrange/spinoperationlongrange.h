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

#ifndef SPINOPERATIONLONGRANGE
#define SPINOPERATIONLONGRANGE

#include "spinoperation.h"

#ifdef WIN32
 #ifdef LONGRANGE_EXPORTS
  #define LONGRANGE_API __declspec(dllexport)
 #else
  #define LONGRANGE_API __declspec(dllimport)
 #endif
#else
 #define LONGRANGE_API 
#endif


#include <complex>
#include <fftw3.h>

using namespace std;

class LongRange : public SpinOperation
{
public:
	LongRange(const char* name, const int field_slot, int nx, int ny, int nz, const int encode_tag);
	virtual ~LongRange();
	
	bool apply(SpinSystem* ss);
	void getMatrices();
	
	virtual void encode(buffer* b)=0;
	virtual int  decode(buffer* b)=0;

	double ABC[9]; //unit cell vectors
	double g; //scale
	int gmax; //longrange cut-off

	virtual void init();
	virtual void deinit();
	
	virtual void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)=0;

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


#endif
