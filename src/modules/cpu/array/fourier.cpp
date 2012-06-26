#include "fourier.h"
#include <fftw3.h>
#define FFT_FORWARD -1
#define FFT_BACKWARD 1

#ifdef WIN32
// BS functions so that windows template exports are happy
void ARRAY_API execute_FFT_PLAN(FFT_PLAN* plan, int* dest, int* src, int* ws) {}
void ARRAY_API execute_FFT_PLAN(FFT_PLAN* plan, float* dest, float* src, float* ws) {}
void ARRAY_API execute_FFT_PLAN(FFT_PLAN* plan, double* dest, double* src, double* ws) {}
#endif


typedef struct FFT_PLAN
{
	int nx, ny, nz;
	int direction;
	int dims;
	
#ifdef DOUBLE_ARRAY
	fftw_plan pd;
#endif	
#ifdef SINGLE_ARRAY
	fftwf_plan pf;
#endif
} FFT_PLAN;


#ifdef DOUBLE_ARRAY
FFT_PLAN* make_FFT_PLAN_double(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	FFT_PLAN* p = new FFT_PLAN;
	p->nx = nx;
	p->ny = ny;
	p->nz = nz;
	p->dims= fftdims;
	p->direction = direction;
	p->pd = 0;
#ifdef SINGLE_ARRAY
	p->pf = 0;
#endif
	
	doubleComplex* a = new doubleComplex[nx*ny*nz];
	doubleComplex* b = new doubleComplex[nx*ny*nz];
	
	fftw_iodim dims[3];
	dims[0].n = nx;
	dims[0].is= 1;
	dims[0].os= 1;
	dims[1].n = ny;
	dims[1].is= nx;
	dims[1].os= nx;
	dims[2].n = nz;
	dims[2].is= nx*ny;
	dims[2].os= nx*ny;
		
	if(fftdims >= 1 && fftdims <= 3)
	{
		p->pd = fftw_plan_guru_dft(fftdims, dims, 0, dims,
								reinterpret_cast<fftw_complex*>(a),
								reinterpret_cast<fftw_complex*>(b),
								direction, FFTW_PATIENT);
	}
	
	delete [] a;
	delete [] b;
	return p;
}
#endif

#ifdef SINGLE_ARRAY
FFT_PLAN* make_FFT_PLAN_float(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	FFT_PLAN* p = new FFT_PLAN;
	p->nx = nx;
	p->ny = ny;
	p->nz = nz;
	p->dims= fftdims;
	p->direction = direction;
	
#ifdef DOUBLE_ARRAY
	p->pd = 0;
#endif
	p->pf = 0;
	
	
	floatComplex* a = new floatComplex[nx*ny*nz];
	floatComplex* b = new floatComplex[nx*ny*nz];
	
	fftw_iodim dims[3];
	dims[0].n = nx;
	dims[0].is= 1;
	dims[0].os= 1;
	dims[1].n = ny;
	dims[1].is= nx;
	dims[1].os= nx;
	dims[2].n = nz;
	dims[2].is= nx*ny;
	dims[2].os= nx*ny;
		
	if(fftdims >= 1 && fftdims <= 3)
	{
		p->pf = fftwf_plan_guru_dft(fftdims, dims, 0, dims,
								reinterpret_cast<fftwf_complex*>(a),
								reinterpret_cast<fftwf_complex*>(b),
								direction, FFTW_PATIENT);
	}
	
	delete [] a;
	delete [] b;
	return p;	
}
#endif


#ifdef SINGLE_ARRAY
void execute_FFT_PLAN(FFT_PLAN* plan, floatComplex* dest, floatComplex* src, floatComplex* ws)
{
	if(!plan->pf) return;
	const int nx = plan->nx;
	const int ny = plan->ny;
	const int nz = plan->nz;
	
	switch(plan->dims)
	{
		case 1:
			for(int i=0; i<ny*nz; i++)
			{
				fftwf_execute_dft(plan->pf, 
					reinterpret_cast<fftwf_complex*>( src + i * nx),
					reinterpret_cast<fftwf_complex*>(dest + i * nx));
			}
		break;
		case 2:
			for(int i=0; i<nz; i++)
			{
				fftwf_execute_dft(plan->pf, 
					reinterpret_cast<fftwf_complex*>( src + i * nx*ny),
					reinterpret_cast<fftwf_complex*>(dest + i * nx*ny));
			}
		break;
		case 3:
			{
				fftwf_execute_dft(plan->pf, 
					reinterpret_cast<fftwf_complex*>( src),
					reinterpret_cast<fftwf_complex*>(dest));
			}
		break;
	}
}
#endif

#ifdef DOUBLE_ARRAY
void execute_FFT_PLAN(FFT_PLAN* plan, doubleComplex* dest, doubleComplex* src, doubleComplex* /* ws */)
{
	if(!plan->pd) return;
	const int nx = plan->nx;
	const int ny = plan->ny;
	const int nz = plan->nz;
	
	switch(plan->dims)
	{
		case 1:
			for(int i=0; i<ny*nz; i++)
			{
				fftw_execute_dft(plan->pd, 
					reinterpret_cast<fftw_complex*>( src + i * nx),
					reinterpret_cast<fftw_complex*>(dest + i * nx));
			}
		break;
		case 2:
			for(int i=0; i<nz; i++)
			{
				fftw_execute_dft(plan->pd, 
					reinterpret_cast<fftw_complex*>( src + i * nx*ny),
					reinterpret_cast<fftw_complex*>(dest + i * nx*ny));
			}
		break;
		case 3:
			{
				fftw_execute_dft(plan->pd, 
					reinterpret_cast<fftw_complex*>( src),
					reinterpret_cast<fftw_complex*>(dest));
			}
		break;
	}
}
#endif


void free_FFT_PLAN(FFT_PLAN* p)
{
	if(!p) return;
#ifdef DOUBLE_ARRAY
	if(p->pd)fftw_destroy_plan(p->pd);
#endif
#ifdef SINGLE_ARRAY
	if(p->pf)fftwf_destroy_plan(p->pf);
#endif
	delete p;
}
