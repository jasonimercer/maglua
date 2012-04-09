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
	LongRange(const char* name="LongRange", const int field_slot=DIPOLE_SLOT, int nx=1, int ny=1, int nz=1, const int encode_tag=hash32("LongRange"));
	virtual ~LongRange();
	
	LINEAGE2("LongRange", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);

	
	
	bool apply(SpinSystem* ss);
	void getMatrices();
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

	double ABC[9]; //unit cell vectors (not shape of sites)
	double g; //scale
	int gmax; //longrange cut-off

	virtual void init();
	virtual void deinit();
	
	virtual void loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ) {};

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
