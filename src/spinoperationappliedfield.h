#ifndef SPINOPERATIONAPPLIEDFIELD
#define SPINOPERATIONAPPLIEDFIELD

#include "spinoperation.h"

class AppliedField : public SpinOperation
{
public:
	AppliedField(int nx, int ny, int nz);
	virtual ~AppliedField();
	
	bool apply(SpinSystem* ss);

	double B[3];
};

AppliedField* checkAppliedField(lua_State* L, int idx);
void registerAppliedField(lua_State* L);


#endif
