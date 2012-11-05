double magnetostatic_Nxx(const double X,   const double Y,   const double Z, const double* target, const double* source);
double magnetostatic_Nxy(const double X,   const double Y,   const double Z, const double* target, const double* source);
double magnetostatic_Nxz(const double X,   const double Y,   const double Z, const double* target, const double* source);

double magnetostatic_Nyx(const double X,   const double Y,   const double Z, const double* target, const double* source);
double magnetostatic_Nyy(const double X,   const double Y,   const double Z, const double* target, const double* source);
double magnetostatic_Nyz(const double X,   const double Y,   const double Z, const double* target, const double* source);

double magnetostatic_Nzx(const double X,   const double Y,   const double Z, const double* target, const double* source);
double magnetostatic_Nzy(const double X,   const double Y,   const double Z, const double* target, const double* source);
double magnetostatic_Nzz(const double X,   const double Y,   const double Z, const double* target, const double* source);

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

void register_mag2d_internal_functions(lua_State* L);
