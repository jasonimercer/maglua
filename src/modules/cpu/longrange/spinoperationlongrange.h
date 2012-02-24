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

class LONGRANGE_API LongRange : public SpinOperation
{
public:
	LongRange(std::string Name, const int field_slot, int nx, int ny, int nz, const int encode_tag);
	virtual ~LongRange();
	
	bool apply(SpinSystem* ss);
	void getMatrices();
	
	virtual void encode(buffer* b)=0;
	virtual int  decode(buffer* b)=0;

	double ABC[9]; //unit cell vectors (not shape of sites)
	double g; //scale
	int gmax; //longrange cut-off

	virtual void init();
	virtual void deinit();
	
	virtual void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)=0;

	double getXX(int ox, int oy, int oz);
	void   setXX(int ox, int oy, int oz, double value);

	double getXY(int ox, int oy, int oz);
	void   setXY(int ox, int oy, int oz, double value);
	
	double getXZ(int ox, int oy, int oz);
	void   setXZ(int ox, int oy, int oz, double value);

	double getYY(int ox, int oy, int oz);
	void   setYY(int ox, int oy, int oz, double value);

	double getYZ(int ox, int oy, int oz);
	void   setYZ(int ox, int oy, int oz, double value);

	double getZZ(int ox, int oy, int oz);
	void   setZZ(int ox, int oy, int oz, double value);
	
	double getAB(int matrix, int ox, int oy, int oz);
	void   setAB(int matrix, int ox, int oy, int oz, double value);
	
private:
	void loadMatrix();

	void ifftAppliedForce(SpinSystem* ss);
	void collectIForces(SpinSystem* ss);

	bool newdata;
	bool hasMatrices;
	bool matrixLoaded;

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

	double* XX;
	double* XY;
	double* XZ;
	double* YY;
	double* YZ;
	double* ZZ;

	fftw_plan forward;
	fftw_plan backward;
};


#endif
