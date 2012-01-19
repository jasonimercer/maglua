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

#ifndef SPINOPERATIONLONGRANGECUDA
#define SPINOPERATIONLONGRANGECUDA

#include "spinoperation.h"
#include "longrange_kernel.hpp"


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef LONGRANGECUDA_EXPORTS
  #define LONGRANGECUDA_API __declspec(dllexport)
 #else
  #define LONGRANGECUDA_API __declspec(dllimport)
 #endif
#else
 #define LONGRANGECUDA_API 
#endif

using namespace std;

class LongRangeCuda : public SpinOperation
{
public:
	LongRangeCuda(const char* name, const int field_slot, int nx, int ny, int nz, const int encode_tag);
	virtual ~LongRangeCuda();
	
	bool apply(SpinSystem* ss);
	bool applyToSum(SpinSystem* ss);
	
	double ABC[9];
	double g;
	int gmax;
	
	virtual void encode(buffer* b)=0;
	virtual int  decode(buffer* b)=0;

	void init();
	void deinit();
	
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
	bool getPlan();
	bool newHostData;
	bool matrixLoaded;
	
	JM_LONGRANGE_PLAN* plan;

	double* XX;
	double* XY;
	double* XZ;
	double* YY;
	double* YZ;
	double* ZZ;
};


#endif
