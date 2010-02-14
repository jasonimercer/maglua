#ifndef SPINOPERATIONTHERMAL
#define SPINOPERATIONTHERMAL

#include "spinoperation.h"

class RNG;
class LLG;
class Thermal : public SpinOperation
{
public:
	Thermal(int nx, int ny, int nz);
	virtual ~Thermal();
	
	bool apply(LLG* llg, RNG* rand, SpinSystem* ss);
	bool apply(SpinSystem* ss) {return false;};

	void scaleSite(int px, int py, int pz, double strength);

	double temperature;
	double* scale;
	
	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);
};

Thermal* checkThermal(lua_State* L, int idx);
void registerThermal(lua_State* L);


#endif
