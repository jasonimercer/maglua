#include "main.h"
#include "luacommon.h"

void registerLibs(lua_State* L)
{
	luaL_openlibs(L);
	registerSpinSystem(L);
	registerLLG(L);
	registerExchange(L);
	registerAppliedField(L);
	registerAnisotropy(L);
	registerDipole(L);
	registerRandom(L);
	registerThermal(L);
	registerConvert(L);
	registerInterpolatingFunction(L);
	registerInterpolatingFunction2D(L);
#ifdef _MPI
	registerMPI(L);
#endif
}
