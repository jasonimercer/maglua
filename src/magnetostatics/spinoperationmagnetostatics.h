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

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#ifndef SPINOPERATIONMAGNETOSTATICS
#define SPINOPERATIONMAGNETOSTATICS

#include "spinoperation.h"

#include <complex>
#include <fftw3.h>

using namespace std;

class Magnetostatic : public SpinOperation
{
	public:
		Magnetostatic(int nx, int ny, int nz);
		virtual ~Magnetostatic();
		
		bool apply(SpinSystem* ss);
		void getMatrices();
		
		double g;
		
		double ABC[9];
		int gmax;
		
		virtual void encode(buffer* b);
		virtual int  decode(buffer* b);
		
		double volumeDimensions[3];
		double crossover_tolerance; //calculations crossover from magnetostatics to dipole
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

void lua_pushMagnetostatic(lua_State* L, Magnetostatic* d);
Magnetostatic* checkMagnetostatic(lua_State* L, int idx);
void registerMagnetostatic(lua_State* L);


#endif
