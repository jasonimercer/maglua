#include <math.h>

double dipole2d_Nxx(const double rx, const double ry, const double rz)
{
	const double r = sqrt(rx*rx+ry*ry+rz*rz);
	if(r < 1E-80)
		return 0;
	const double ir = 1.0 / r;
	const double ir3 = ir*ir*ir;
	const double ir5=ir3*ir*ir;
	return  -1.0 * (ir3 - 3.0 * rx * rx * ir5);
}
double dipole2d_Nxy(const double rx, const double ry, const double rz)
{
	const double r = sqrt(rx*rx+ry*ry+rz*rz);
	if(r < 1E-80)
		return 0;
	const double ir = 1.0 / r;
	const double ir3 = ir*ir*ir;
	const double ir5=ir3*ir*ir;
	return  -1.0 * (         - 3.0 * rx * ry * ir5);
}
double dipole2d_Nxz(const double rx, const double ry, const double rz)
{
	return dipole2d_Nxy(rx, rz, ry);
}



double dipole2d_Nyx(const double rx, const double ry, const double rz)
{
	return dipole2d_Nxy(ry, rx, rz);
}
double dipole2d_Nyy(const double rx, const double ry, const double rz)
{
	return dipole2d_Nxx(ry, rx, rz);
}
double dipole2d_Nyz(const double rx, const double ry, const double rz)
{
	return dipole2d_Nxy(ry, rz, rx);
}


double dipole2d_Nzx(const double rx, const double ry, const double rz)
{
	return dipole2d_Nxy(rz, rx, ry);
}
double dipole2d_Nzy(const double rx, const double ry, const double rz)
{
	return dipole2d_Nxy(rz, ry, rx);
}
double dipole2d_Nzz(const double rx, const double ry, const double rz)
{
	return dipole2d_Nxx(rz, rx, ry);
}




double dipole2d_Nxx_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nxx(xx,yy,rz);
		}
	}
	return sum;
}

double dipole2d_Nxy_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nxy(xx,yy,rz);
		}
	}
	return sum;
}

double dipole2d_Nxz_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nxz(xx,yy,rz);
		}
	}
	return sum;
}




double dipole2d_Nyx_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nxx(xx,yy,rz);
		}
	}
	return sum;
}

double dipole2d_Nyy_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nyy(xx,yy,rz);
		}
	}
	return sum;
}

double dipole2d_Nyz_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nyz(xx,yy,rz);
		}
	}
	return sum;
}





double dipole2d_Nzx_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nzx(xx,yy,rz);
		}
	}
	return sum;
}

double dipole2d_Nzy_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nzy(xx,yy,rz);
		}
	}
	return sum;
}

double dipole2d_Nzz_range(const double rx, const double ry, const double rz, 
						  const double sx, const double sy,
						  const int nx, const int ny)
{
	long double sum = 0;
	for(int x=-nx; x<=nx; x++)
	{
		for(int y=-ny; y<=ny; y++)
		{
			const double xx = x*sx + rx;
			const double yy = y*sy + ry;
			sum += dipole2d_Nzz(xx,yy,rz);
		}
	}
	return sum;
}



