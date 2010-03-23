#ifndef SPINOPERATION
#define SPINOPERATION

#define SUM_SLOT          0
#define EXCHANGE_SLOT     1
#define APPLIEDFIELD_SLOT 2
#define ANISOTROPY_SLOT   3
#define THERMAL_SLOT      4
#define DIPOLE_SLOT       5
#define NSLOTS            6

//#include <omp.h>
#include "luacommon.h"
#include <string>
#include "encodable.h"
class SpinSystem;

class SpinOperation : public Encodable
{
public:
	SpinOperation(std::string Name, int slot, int nx, int ny, int nz, int encodetype);
	virtual ~SpinOperation();
	
	virtual bool apply(SpinSystem* ss) = 0;
	int getSite(int x, int y, int z);

	bool member(int px, int py, int pz);
	int  getidx(int px, int py, int pz);

	int nx, ny, nz;
	int nxyz;
	const std::string& name();
	int refcount;

	std::string errormsg;
	
	virtual void encode(buffer* b) const = 0;
	virtual int  decode(buffer* b) = 0;

protected:
	std::string operationName;
	int slot;
};

#endif
