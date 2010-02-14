#include "spinoperation.h"
#define CLAMP(x, m) ((x<0)?0:(x>m?m:x))

using namespace std;

SpinOperation::SpinOperation(std::string Name, int Slot, int NX, int NY, int NZ)
	: operationName(Name), slot(Slot), refcount(0), nx(NX), ny(NY), nz(NZ) 
{
	nxyz = nx * ny * nz;
}

SpinOperation::~SpinOperation()
{
	
}

const string& SpinOperation::name()
{
	return operationName;
}
	
int SpinOperation::getSite(int x, int y, int z)
{
	x = (x+10*nx) % nx;
	y = (y+10*ny) % ny;
	z = (z+10*nz) % nz;

	return x + nx*y + nx*ny*z;
}

bool SpinOperation::member(int px, int py, int pz)
{
	if(px < 0 || py < 0 || pz < 0)
		return false;

	if(px >= nx || py >= ny || pz >= nz)
		return false;
	
	return true;
}

int  SpinOperation::getidx(int px, int py, int pz)
{
	px = CLAMP(px, nx);
	py = CLAMP(py, ny);
	pz = CLAMP(pz, nz);
	
	return px + nx * (py + ny * pz);
}
