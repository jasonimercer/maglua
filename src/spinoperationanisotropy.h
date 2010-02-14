#ifndef SPINOPERATIONANISOTROPY
#define SPINOPERATIONANISOTROPY

#include "spinoperation.h"

class Anisotropy : public SpinOperation
{
public:
	Anisotropy(int nx, int ny, int nz);
	virtual ~Anisotropy();
	
	bool apply(SpinSystem* ss);

	double* ax;
	double* ay;
	double* az;
	double* strength;
	
	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);

};

Anisotropy* checkAnisotropy(lua_State* L, int idx);
void registerAnisotropy(lua_State* L);


#endif
