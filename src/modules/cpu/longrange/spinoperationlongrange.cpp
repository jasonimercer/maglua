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

#include "spinsystem.h"
#include "info.h"
#include "spinoperationlongrange.h"

#include <stdlib.h>
#include <math.h>

LongRange::LongRange(const char* name, const int field_slot, int nx, int ny, int nz, const int encode_tag)
	: SpinOperation(name, field_slot, nx, ny, nz, encode_tag)
{
	qXX = 0;
	
	forward = 0;
        backward = 0;

	g = 1;
	gmax = 2000;
	matrixLoaded = false;
	newdata = true;
	XX = 0;
	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	hasMatrices = false;
}

void LongRange::init()
{
	if(XX) return;
	
	deinit();
	int s = nx*ny * nz;

	hqx = new complex<double> [s];
	hqy = new complex<double> [s];
	hqz = new complex<double> [s];

	hrx = new complex<double> [s];
	hry = new complex<double> [s];
	hrz = new complex<double> [s];

	qXX = new complex<double>[s];
	qXY = new complex<double>[s];
	qXZ = new complex<double>[s];

	qYY = new complex<double>[s];
	qYZ = new complex<double>[s];
	qZZ = new complex<double>[s];
	
	fftw_iodim dims[2];
	dims[0].n = nx;
	dims[0].is= 1;
	dims[0].os= 1;
	dims[1].n = ny;
	dims[1].is= nx;
	dims[1].os= nx;

	forward = fftw_plan_guru_dft(2, dims, 0, dims,
								reinterpret_cast<fftw_complex*>(qXX),
								reinterpret_cast<fftw_complex*>(qYY),
								FFTW_FORWARD, FFTW_PATIENT);

	backward= fftw_plan_guru_dft(2, dims, 0, dims,
								reinterpret_cast<fftw_complex*>(qXX),
								reinterpret_cast<fftw_complex*>(qYY),
								FFTW_BACKWARD, FFTW_PATIENT);
								
 	XX = new double[s];
	XY = new double[s];
	XZ = new double[s];
	YY = new double[s];
	YZ = new double[s];
	ZZ = new double[s];
	
	hasMatrices = false;
}

static int offsetOK(int nx, int ny, int nz,  int x, int y, int z, int& offset)
{
	if(x<0 || x >= nx) return 0;
	if(y<0 || y >= ny) return 0;
	if(z<0 || z >= nz) return 0;
	
	offset = x + y*nx + z*nx*ny;
	return 1;
}

#define getsetPattern(AB) \
double LongRange::get##AB (int ox, int oy, int oz) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
    loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
		return AB [offset]; \
	return 0; \
} \
 \
void   LongRange::set##AB (int ox, int oy, int oz, double value) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
	loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
	{ \
		AB [offset] = value; \
		newdata = true; \
	} \
} 

getsetPattern(XX)
getsetPattern(XY)
getsetPattern(XZ)
getsetPattern(YY)
getsetPattern(YZ)
getsetPattern(ZZ)

double LongRange::getAB(int matrix, int ox, int oy, int oz)
{
	switch(matrix)
	{
		case 0:	return getXX(ox,oy,oz);
		case 1:	return getXY(ox,oy,oz);
		case 2:	return getXZ(ox,oy,oz);
		case 3:	return getYY(ox,oy,oz);
		case 4:	return getYZ(ox,oy,oz);
		case 5:	return getZZ(ox,oy,oz);
	}
	return 0;
}

void  LongRange::setAB(int matrix, int ox, int oy, int oz, double value)
{
	switch(matrix)
	{
		case 0:	setXX(ox,oy,oz,value); break;
		case 1:	setXY(ox,oy,oz,value); break;
		case 2:	setXZ(ox,oy,oz,value); break;
		case 3:	setYY(ox,oy,oz,value); break;
		case 4:	setYZ(ox,oy,oz,value); break;
		case 5:	setZZ(ox,oy,oz,value); break;
	}
}

void LongRange::loadMatrix()
{
	if(matrixLoaded) return;
	init();
	loadMatrixFunction(XX, XY, XZ, YY, YZ, ZZ); //implemented by child
	newdata = true;
	matrixLoaded = true;
}

void LongRange::deinit()
{
	if(qXX)
	{
		delete [] qXX;
		delete [] qXY;
		delete [] qXZ;

		delete [] qYY;
		delete [] qYZ;

		delete [] qZZ;

		delete [] hqx;
		delete [] hqy;
		delete [] hqz;

		delete [] hrx;
		delete [] hry;
		delete [] hrz;

		fftw_destroy_plan(forward);
		fftw_destroy_plan(backward);
		qXX = 0;
	}
	if(XX)
	{
		delete [] XX;
		delete [] XY;
		delete [] XZ;
		delete [] YY;
		delete [] YZ;
		delete [] ZZ;
		XX = 0;
	}
	
	hasMatrices = false;
}

LongRange::~LongRange()
{
	deinit();
}

void LongRange::getMatrices()
{
	init();
	
	loadMatrix();

	fftw_iodim dims[2];
	dims[0].n = nx;
	dims[0].is= 1;
	dims[0].os= 1;
	dims[1].n = ny;
	dims[1].is= nx;
	dims[1].os= nx;

	complex<double>* r = new complex<double>[nx*ny];
	complex<double>* q = new complex<double>[nx*ny];
	
	double* arrs[6];
	arrs[0] = XX;
	arrs[1] = XY;
	arrs[2] = XZ;
	arrs[3] = YY;
	arrs[4] = YZ;
	arrs[5] = ZZ;
	
	complex<double>* qarrs[6];
	qarrs[0] = qXX;
	qarrs[1] = qXY;
	qarrs[2] = qXZ;
	qarrs[3] = qYY;
	qarrs[4] = qYZ;
	qarrs[5] = qZZ;
	
	for(int a=0; a<6; a++)
		for(int k=0; k<nz; k++)
		{
			for(int i=0; i<nx*ny; i++)
				r[i] = complex<double>(arrs[a][k*nx*ny + i],0);
		
			fftw_execute_dft(forward, 
					reinterpret_cast<fftw_complex*>(r),
					reinterpret_cast<fftw_complex*>(q));

			for(int i=0; i<nx*ny; i++)
				qarrs[a][k*nx*ny + i] = q[i];
		}
	
	
	
	delete [] q;
	delete [] r;

	newdata = false;
	hasMatrices = true;
}

void LongRange::ifftAppliedForce(SpinSystem* ss)
{
	double d = g / (double)(nx*ny);
// 	printf("%g\n", d);
	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];
	const int nxy = nx*ny;
	
	for(int i=0; i<nz; i++)
	{
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqx[i*nxy]),
				reinterpret_cast<fftw_complex*>(&hrx[i*nxy]));
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqy[i*nxy]),
				reinterpret_cast<fftw_complex*>(&hry[i*nxy]));
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqz[i*nxy]),
				reinterpret_cast<fftw_complex*>(&hrz[i*nxy]));
	}

	for(int i=0; i<nxyz; i++)
		hx[i] = hrx[i].real() * d;

	for(int i=0; i<nxyz; i++)
		hy[i] = hry[i].real() * d;

	for(int i=0; i<nxyz; i++)
		hz[i] = hrz[i].real() * d;
}


void LongRange::collectIForces(SpinSystem* ss)
{
	int c;
	int sourceLayer, targetLayer;// !!Source layer, Target Layer
	int sourceOffset;
	int targetOffset;
	int demagOffset;
	const int nxy = nx*ny;
	
	complex<double>* sqx = ss->qx;
	complex<double>* sqy = ss->qy;
	complex<double>* sqz = ss->qz;

// 	if(!hasMatrices)
// 		getMatrices();
	
	const complex<double> cz(0,0);
	for(c=0; c<nxyz; c++) hqx[c] = cz;
	for(c=0; c<nxyz; c++) hqy[c] = cz;
	for(c=0; c<nxyz; c++) hqz[c] = cz;

	
# define cSo c+sourceOffset
# define cDo c+ demagOffset
# define cTo c+targetOffset


	for(targetLayer=0; targetLayer<nz; targetLayer++)
	for(sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		int offset = sourceLayer - targetLayer;
		double sign = 1.0;
		if(offset < 0)
		{
			offset = -offset;
			sign = -sign;
		}
	
		targetOffset = targetLayer * nxy;
		sourceOffset = sourceLayer * nxy;
		demagOffset  = offset * nxy;
		//these are complex multiplies and adds
		for(c=0; c<nxy; c++) hqx[cTo]+=qXX[cDo]*sqx[cSo];
		for(c=0; c<nxy; c++) hqx[cTo]+=qXY[cDo]*sqy[cSo];
		for(c=0; c<nxy; c++) hqx[cTo]+=qXZ[cDo]*sqz[cSo]*sign;

		for(c=0; c<nxy; c++) hqy[cTo]+=qXY[cDo]*sqx[cSo];
		for(c=0; c<nxy; c++) hqy[cTo]+=qYY[cDo]*sqy[cSo];
		for(c=0; c<nxy; c++) hqy[cTo]+=qYZ[cDo]*sqz[cSo]*sign;

		for(c=0; c<nxy; c++) hqz[cTo]+=qXZ[cDo]*sqx[cSo]*sign;
		for(c=0; c<nxy; c++) hqz[cTo]+=qYZ[cDo]*sqy[cSo]*sign;
		for(c=0; c<nxy; c++) hqz[cTo]+=qZZ[cDo]*sqz[cSo];
	}
}

	
bool LongRange::apply(SpinSystem* ss)
{
	markSlotUsed(ss);

	if(newdata)
		getMatrices();
	
	ss->fft();
	collectIForces(ss);
	ifftAppliedForce(ss);

	return true;
}



extern "C"
{
LONGRANGE_API int lib_register(lua_State* L);
LONGRANGE_API int lib_version(lua_State* L);
LONGRANGE_API const char* lib_name(lua_State* L);
LONGRANGE_API int lib_main(lua_State* L, int argc, char** argv);
}

LONGRANGE_API int lib_register(lua_State* L)
{
	return 0;
}

LONGRANGE_API int lib_version(lua_State* L)
{
	return __revi;
}

LONGRANGE_API const char* lib_name(lua_State* L)
{
	return "LongRange";
}

LONGRANGE_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}



