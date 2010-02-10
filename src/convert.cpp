#include "convert.h"
#include <string.h>
#include <iostream>
using namespace std;

#define UNKNOWN       0
#define KELVIN        1
#define EV            2
#define CELSIUS       3
#define FAHRENHEIT    4
#define GAUSS         5
#define TESLA         6
#define mEV           7
#define ERG           8

typedef struct convrules
{
	int ft[2];
	double ama[3];
}convrules;

typedef struct compoundconv
{
	int ft[2];
	int n;
	int steps[6];
}compoundconv;

const convrules rules[] =
{
	{{KELVIN,       CELSIUS},      {-273.15,     1.0,    0.0}},
	{{FAHRENHEIT,   CELSIUS},      {    -32, 5.0/9.0,    0.0}},
	{{GAUSS,        TESLA},        {      0,  0.0001,    0.0}},
	{{mEV,          EV},           {      0,   0.001,    0.0}},
	{{ERG,          EV},           {      0, 1.6E-12,    0.0}},
	{{0,0}, {0,0,0}}
};

const compoundconv crules[] =
{
	{{FAHRENHEIT, KELVIN}, 2, {FAHRENHEIT, CELSIUS, KELVIN}},
	{{0,0}}
};

#define is(n)  (strcasecmp(name, n) == 0)
#define isc(n) (strcmp(name, n) == 0)
int unitType(const char* name)
{
	if(is("kelvin") || is("k"))
		return KELVIN;
	
	if(is("ElectronVolts") || is("Electron Volts") || is("eV"))
		return EV;
	
	if(is("Celsius") || is("Celcius") || is("C"))
		return CELSIUS;
	
	if(is("fahrenheit") || is("farenheit"))
		return FAHRENHEIT;

	if(is("Gauss"))
		return GAUSS;

	if(is("Tesla"))
		return TESLA;

	if(isc("meV") || isc("mEV") || is("millielectronvolts"))
		return mEV;
	
	if(is("erg"))
		return ERG;
	
	// 8.6173x10^-5 eV/K  (electronvolts per kelvin)
	
	return UNKNOWN;
}

int _conv(int fromType, int toType, double value, double* conv)
{
	if(fromType == toType)
	{
		*conv = value;
		return 1;
	}
	int i = 0;
	while(rules[i].ft[0])
	{
		if(rules[i].ft[0] == fromType && rules[i].ft[1] == toType)
		{
			*conv = ((value + rules[i].ama[0]) * rules[i].ama[1]  + rules[i].ama[2]);
			return 1;
		}

		if(rules[i].ft[1] == fromType && rules[i].ft[0] == toType)
		{
			*conv = (value - rules[i].ama[2]) / rules[i].ama[1] - rules[i].ama[0];
			return 1;
		}

		i++;
	}
	
	i = 0; //check compound rules
	while(crules[i].ft[0])
	{
		if(crules[i].ft[0] == fromType && crules[i].ft[1] == toType)
		{
			int n = crules[i].n;
			for(int j=0; j<n; j++)
			{
				if(!_conv(crules[i].steps[j], crules[i].steps[j+1], value, conv))
					return 0;
				value = *conv;
			}
			return 1;
		}

		if(crules[i].ft[1] == fromType && crules[i].ft[0] == toType)
		{
			int n = crules[i].n;
			for(int j=n-1; j>=0; j++)
			{
				if(!_conv(crules[i].steps[j+1], crules[i].steps[j], value, conv))
					return 0;
				value = *conv;
			}
			return 1;
		}
		i++;
	}
	
	return 0;
}

static int l_convert(lua_State* L)
{
	int n = lua_gettop(L);
	
	if(n < 2)
		return luaL_error(L, "convert requires from and to units. Ex: convert(5 , \"K\", \"eV\")");
	
	const char* fromUnits = lua_tostring(L, -2);
	const char* toUnits   = lua_tostring(L, -1);
	
	int fu = unitType(fromUnits);
	int tu = unitType(toUnits);
	
	if(!fu)
		return luaL_error(L, "Failed to determine type of `%s'", fromUnits);
	
	if(!tu)
		return luaL_error(L, "Failed to determine type of `%s'", toUnits);
	
	int v = 0;
	for(int i=0; i<n-2; i++)
	{
		double d;
		
		if(!_conv(fu, tu, lua_tonumber(L, i+1), &d))
			return luaL_error(L, "No rules to convert from `%s' to `%s'", fromUnits, toUnits);
		
		lua_pushnumber(L, d);
		v++;
	}
	
	return v;
}




void registerConvert(lua_State* L)
{
	lua_pushcfunction(L, l_convert);
	lua_setglobal(L, "convert");
}

