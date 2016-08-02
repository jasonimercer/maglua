#include "UnitCell.h"
#include <math.h>
#include <iostream>
#include <string.h>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

UnitCell::UnitCell()
{
	_A = Vector(1,0,0);
	_B = Vector(0,1,0);
	_C = Vector(0,0,1);
	refcount = 0;
	
	g2r.makeIdentity();
	r2g.makeIdentity();
}

// UnitCell::UnitCell(const Vector& a, const Vector& b, const Vector& c)
// {
// 	setA(a);
// 	setB(b);
// 	setC(c);
// 	refcount = 0;
// }

UnitCell::~UnitCell()
{
	for(unsigned int i=0; i<atoms.size(); i++)
	{
		atoms[i]->refcount--;
		if(atoms[i]->refcount == 0)
			delete atoms[i];
	}
}

void UnitCell::translate(int rx, int ry, int rz)
{
	Vector v(rx,ry,rz);
	v = reducedToGlobal(v);
	
	for(unsigned int i=0; i<atoms.size(); i++)
	{
		atoms[i]->setPos(atoms[i]->pos() + v);
	}
}


void UnitCell::anglesLengthsToBasisVectors(double alpha, double beta, double gamma, double a, double b, double c)
{
	alpha = alpha * M_PI / 180.0;
	beta  = beta  * M_PI / 180.0;
	gamma = gamma * M_PI / 180.0;

	r2g.makeIdentity();
	
	r2g.setComponent(0,0,a);
	r2g.setComponent(0,1,b * cos(gamma));
	r2g.setComponent(0,2,c * cos(beta));

	r2g.setComponent(1,1,b * sin(gamma));
	r2g.setComponent(1,2,c * ((cos(alpha) - cos(beta) * cos(gamma))/sin(gamma)));
	
	const double v = 
	sqrt(1.0 - cos(alpha)*cos(alpha) - cos(beta)*cos(beta) - cos(gamma)*cos(gamma) + 2 * cos(alpha) * cos(beta) * cos(gamma));
	
	r2g.setComponent(2,2,c * v / sin(gamma));
	
	g2r = r2g;
	g2r.invert();
	
	_A = r2g * Vector(1,0,0);
	_B = r2g * Vector(0,1,0);
	_C = r2g * Vector(0,0,1);
	
// 	_A = Vector(a, 0, 0);
// 	_B = Vector(b*cos(gamma), b*sin(gamma), 0);
// 	_C = Vector(cos(beta), (cos(alpha) - cos(beta) * cos(gamma)) / sin(beta), 0);
// 	_C.setZ( sqrt(1.0 - _C.x() * _C.x() - _C.y() * _C.y()) );
// 	_C *= c;
// 
// 	calcInvMatrix();
}

Vector UnitCell::reducedToGlobal(const Vector& v) const
{
	return r2g * v;
// 	return Vector(A().dot(v), B().dot(v), C().dot(v));
// 	return v.x() * A() + v.y() * B() + v.z() * C();
}

Vector UnitCell::globalToReduced(const Vector& v) const
{
	return g2r * v;
//	return Vector(iA.dot(v), iB.dot(v), iC.dot(v));
// 	return v.x()*iA + v.y()*iB + v.z()*iC;
}

	
static char lcase(char c)
{
	if(c<='Z' && c>='A')
		return c-('Z'-'z');
	return c;
} 

bool UnitCell::applyOperator(lua_State* L, int func_idx)
{
	if(!lua_isfunction(L, func_idx))
	{
		luaL_error(L, "Apply operator requires a function");
		return false;
	}
	
	vector<Vector> pp;
	
	for(unsigned int i=0; i<atoms.size(); i++)
	{
		Vector p = atoms[i]->pos();
		p = globalToReduced(p);
		
		lua_pushvalue(L, func_idx); //make a copy of the function

		lua_pushnumber(L, p.x());
		lua_pushnumber(L, p.y());
		lua_pushnumber(L, p.z());
		
		if(lua_pcall(L, 3, 3, 0))
		{
			return false;
		}

		double x = lua_tonumber(L, -3);
		double y = lua_tonumber(L, -2);
		double z = lua_tonumber(L, -1);
		lua_pop(L, 3);

		p = reducedToGlobal(Vector(x, y, z));
		
		pp.push_back(p);
	}
	
	for(unsigned int i=0; i<atoms.size(); i++)
	{
		atoms[i]->setPos(pp[i]);
	}
	
	return true;
}

/* adding code here to make sure the resulting x,y,z values is in [0:1] 
 * for instance, -1/2-x should really be -1/2-x+1 
 */
bool UnitCell::applyOperator(const char* xyz)
{
	lua_State* L = lua_open();
	luaL_openlibs(L);
	
	char* operation = new char[strlen(xyz)+1];
	int n = strlen(xyz);
	for(int i=0; i<n; i++)
	{
		operation[i] = lcase(xyz[i]);
	}
	operation[n] = 0;
	
	char* cmd = new char[strlen(xyz) + 1024];
	
	sprintf(cmd, "function op(x, y, z) return %s end", operation);

	if(luaL_dostring(L, cmd))
	{
		lua_close(L);
		return false;
	}
	
	lua_getglobal(L, "op");
	
	/* test for data range */
	{
		lua_pushvalue(L, -1);
		lua_pushnumber(L, 0.5);
		lua_pushnumber(L, 0.5);
		lua_pushnumber(L, 0.5);
		if(lua_pcall(L, 3,3,0))
		{
			lua_close(L);
			return false;
		}
		double xyz[3];
		xyz[0] = lua_tonumber(L, -3);
		xyz[1] = lua_tonumber(L, -2);
		xyz[2] = lua_tonumber(L, -1);
		
		double d[3] = {0,0,0};

		for(int i=0; i<3; i++)
		{
			while((xyz[i]+d[i]) < 0.25) d[i]++;
			while((xyz[i]+d[i]) > 0.75) d[i]--;
		}
		
		sprintf(cmd, "function op(x, y, z)\n\ta,b,c = %s\n\treturn a+%f,b+%f,c+%f end\n", operation, d[0], d[1], d[2]);
		if(luaL_dostring(L, cmd))
		{
			lua_close(L);
			return false;
		}
		lua_pop(L, lua_gettop(L));
		lua_getglobal(L, "op");
	}
	
	bool r = applyOperator(L, lua_gettop(L));
	
	lua_close(L);
	return r;
}


void UnitCell::addAtomGlobalCoorinates(Atom* a)
{
	a->refcount++;
	atoms.push_back(a);
}

void UnitCell::addAtomReducedCoorinates(Atom* a)
{
// 	Vector p = a->pos();
// 	Vector d = reducedToGlobal(p);
	a->setPos(reducedToGlobal(a->pos()));
	
	
// 	a->setPos(reducedToGlobal(a->pos()));
	addAtomGlobalCoorinates(a);
}




Vector UnitCell::A() const
{
	return _A;
}

Vector UnitCell::B() const
{
	return _B;
}

Vector UnitCell::C() const
{
	return _C;
}


void UnitCell::setA(const Vector& v)
{
	_A = v;
}

void UnitCell::setB(const Vector& v)
{
	_B = v;
}

void UnitCell::setC(const Vector& v)
{
	_C = v;
}








int lua_isunitcell(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "UnitCell");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}


UnitCell* lua_tounitcell(lua_State* L, int idx)
{
	UnitCell** pp = (UnitCell**)luaL_checkudata(L, idx, "UnitCell");
	luaL_argcheck(L, pp != NULL, 1, "`UnitCell' expected");
	return *pp;
}

void lua_pushunitcell(lua_State* L, UnitCell* u)
{
	UnitCell** pp = (UnitCell**)lua_newuserdata(L, sizeof(UnitCell**));

	*pp = u;
	luaL_getmetatable(L, "UnitCell");
	lua_setmetatable(L, -2);
	u->refcount++;
}

static int l_uc_new(lua_State* L)
{
	if(lua_gettop(L))
	{
		return luaL_error(L, "UnitCell.new doesn't take parameters");
	}
// 		Vector a, b, c;
// 		int idx = 1;
// 		idx += lua_makevector(L, idx, a);
// 		idx += lua_makevector(L, idx, b);
// 		idx += lua_makevector(L, idx, c);
// 		
// 		lua_pushunitcell(L, new UnitCell(a,b,c));
// 	}
// 	else
	lua_pushunitcell(L, new UnitCell);
	
	return 1;
}

static int l_uc_gc(lua_State* L)
{
	UnitCell* c = lua_tounitcell(L, 1);
	if(!c) return 0;
	
	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_uc_al2bv(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;
	
	double v[6];
	for(int i=0; i<6; i++)
	{
		v[i] = lua_tonumber(L, i+2);
		if(v[i] == 0)
			return luaL_error(L, "UnitCell:anglesLengthsToBasisVectors requires 6 numbers");
	}
	
	uc->anglesLengthsToBasisVectors(
		v[0], v[1], v[2],
		v[3], v[4], v[5]);
	return 0;
}
	
static int l_uc_ggc(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;

	Vector v;
	lua_makevector(L, 2, v);
	
	Vector b = uc->reducedToGlobal(v);
	lua_pushvector(L, new Vector(b));
	return 1;
}

static int l_uc_g2r(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;
	
	Vector v;
	lua_makevector(L, 2, v);

	Vector b = uc->globalToReduced(v);
	lua_pushvector(L, new Vector(b));
	return 1;
}

static int l_uc_A(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;

	lua_pushvector(L, new Vector(uc->A()));
	return 1;
}
static int l_uc_B(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;

	lua_pushvector(L, new Vector(uc->B()));
	return 1;
}
static int l_uc_C(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;

	lua_pushvector(L, new Vector(uc->C()));
	return 1;
}

static int l_uc_addatomglobal(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;
	
	Atom* a = lua_toatom(L, 2);
	if(!a) return 0;
	
	uc->addAtomGlobalCoorinates(a);
	return 0;
}
static int l_uc_addatomreduced(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;
	
	Atom* a = lua_toatom(L, 2);
	if(!a) return 0;
	
	uc->addAtomReducedCoorinates(a);
	return 0;
}

static int l_uc_applyoperator(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;

	if(lua_isfunction(L, 2))
	{
		uc->applyOperator(L, 2);
		return 0;
	}
	if(lua_isstring(L, 2))
	{
		if(!uc->applyOperator(lua_tostring(L, 2)))
			return luaL_error(L, "Failed to apply `%s'\n", lua_tostring(L, 2));
	}
	return 0;
}

static int l_uc_atomcount(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;
	
	lua_pushinteger(L, uc->atoms.size());
	return 1;
}
static int l_uc_atomat(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;

	int i = lua_tointeger(L, 2) - 1;
	if(i<0) return 0;
	if(i>=uc->atoms.size()) return 0;
	
	lua_pushatom(L, uc->atoms[i]);
	return 1;
}
	
static int l_uc_translate(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;
	
	int x = lua_tointeger(L, 2);
	int y = lua_tointeger(L, 3);
	int z = lua_tointeger(L, 4);

	uc->translate(x, y, z);
	return 0;
}

static int l_uc_copy(lua_State* L)
{
	UnitCell* uc = lua_tounitcell(L, 1);
	if(!uc) return 0;
	
	UnitCell* uc2 = new UnitCell();//uc->A(), uc->B(), uc->C());
	
	uc2->r2g = uc->r2g;
	uc2->g2r = uc->g2r;
	
	uc2->setA(uc->A());
	uc2->setB(uc->B());
	uc2->setC(uc->C());
	
	for(unsigned int i=0; i<uc->atoms.size(); i++)
	{
		uc2->addAtomGlobalCoorinates(new Atom(*uc->atoms[i]));
	}
	lua_pushunitcell(L, uc2);
	return 1;
}

	
void lua_registerunitcell(lua_State* L)
{
	static const struct luaL_reg struct_m [] = { //methods
		{"__gc",       l_uc_gc},
		{"getGlobalCoordinates", l_uc_ggc},
		{"anglesLengthsToBasisVectors", l_uc_al2bv},
		{"addAtomGlobalCoorinates", l_uc_addatomglobal},
		{"addAtomReducedCoorinates", l_uc_addatomreduced},
		{"applyOperator",          l_uc_applyoperator},
		{"atomCount",              l_uc_atomcount},
		{"atomAt",                 l_uc_atomat},
		{"copy",                   l_uc_copy},
		{"translate",              l_uc_translate},
		{"reducedToGlobal",        l_uc_ggc},
		{"globalToReduced",        l_uc_g2r},
		{"A", l_uc_A},
		{"B", l_uc_B},
		{"C", l_uc_C},
		{NULL, NULL}
	};

	luaL_newmetatable(L, "UnitCell");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
			{"new", l_uc_new},
			{NULL, NULL}
	};

	luaL_register(L, "UnitCell", struct_f);
	lua_pop(L,1);
}


