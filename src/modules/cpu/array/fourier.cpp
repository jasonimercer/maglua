#include "fourier.h"
#include <fftw3.h>
#define FFT_FORWARD -1
#define FFT_BACKWARD 1

typedef struct FFT_PLAN
{
	int nx, ny, nz;
	int direction;
	int dims;
	fftw_plan pd;
	fftwf_plan pf;
} FFT_PLAN;


FFT_PLAN* make_FFT_PLAN_double(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	FFT_PLAN* p = new FFT_PLAN;
	p->nx = nx;
	p->ny = ny;
	p->nz = nz;
	p->dims= fftdims;
	p->direction = direction;
	p->pd = 0;
	p->pf = 0;
	
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

FFT_PLAN* make_FFT_PLAN_float(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	FFT_PLAN* p = new FFT_PLAN;
	p->nx = nx;
	p->ny = ny;
	p->nz = nz;
	p->dims= fftdims;
	p->direction = direction;
	
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



void free_FFT_PLAN(FFT_PLAN* p)
{
	if(!p) return;
	if(p->pd)fftw_destroy_plan(p->pd);
	if(p->pf)fftwf_destroy_plan(p->pf);
	delete p;
}
