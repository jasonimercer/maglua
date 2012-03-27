#include "VolumeLua.h"
#include "Atom.h"
#include <math.h>

#include <iostream>
using namespace std;

VolumeLua::VolumeLua()
{
	center = Vector(0,0,0);
	refcount = 0;
	L = 0;
	funcref = 0;
}


VolumeLua::~VolumeLua()
{
	if(L)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, funcref);
	}
}

void VolumeLua::setFunction(lua_State* _L, int _funcref)
{
	if(L)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, funcref);
	}
	
	funcref = _funcref;
	L = _L;
}


bool VolumeLua::rayIntersect(const Ray& ray, double& t)
{
	return false;
}

bool VolumeLua::contains(const Vector& v, double expand)
{
	// a volume is made up of a bunch of tets. 
	// tet face 0 is the outter face
	// a point/sphere is completely contained if it is
	// behind all outter faces
	
	if(bb.excludes(v,expand))
		return false;
		
	for(unsigned int i=0; i<tetrahedron.size(); i++)
	{
		if(tetrahedron[i].tri[0].value(v) > -expand)
			return false;
	}
	return true;
}

bool VolumeLua::excludes(const Vector& v, double expand)
{
	if(bb.excludes(v,expand))
	{
// 		printf("BB excludes\n");
		return true;
	}

	for(unsigned int i=0; i<tetrahedron.size(); i++)
	{
		if(!tetrahedron[i].excludes(v,expand))
			return false;
	}
	
// 	printf("fine exclude\n");
	
	return true;
}


bool VolumeLua::excludes(GroupNode* gn)
{
	if(!gn)
	{
		//printf("empty group\n");
		return true;
	}
	
	if(bb.excludes(gn->bb))
	{
// 		printf("skipping over %i\n", gn->size);
		return true;
	}
// 	cout << gn->bb.min << " -- " << gn->bb.max << endl;
	if(gn->n1) //then children
	{
// 		printf("Checking child\n");
		return excludes(gn->n1) && excludes(gn->n2);
	}
	// data
	//printf("checking data\n");
// 	cout << gn->pos << "  --  " << gn->radius << endl;

	Atom* a = dynamic_cast<Atom*>(gn->v);
	if(!a)
	{
		cerr << "cannot evaluate exclude on non-atom" << endl;
		return false;
	}

	return excludes(a->pos(), a->radius());
}

bool VolumeLua::excludes(Group* g)
{
	if(g->compiled)
		return excludes(g->root);
	
	for(list<GroupNode*>::iterator it=g->precompiled.begin();
		it != g->precompiled.end(); 
		it++)
// 	for(unsigned int i=0; i<g->precompiled.size(); i++)
	{
		Atom* a = dynamic_cast<Atom*>((*it)->v);
		if(!a)
		{
			cerr << "cannot evaluate exclude on non-atom" << endl;
			return false;
		}
		if(!excludes(a->pos(), a->radius()))
			return false;
	}
	return true;
}



void VolumeLua::updateBoundingBox()
{
}

bool VolumeLua::sphereEnclosed(const Sphere& s) const
{
	const Vector& center = s.pos();
	const double rad = s.radius();

	//check for bounding box first
	if(!bb.contains(center))
		return false;
// 	if(center.x() < aabb[0]) return false;
// 	if(center.x() > aabb[1]) return false;
// 	if(center.y() < aabb[2]) return false;
// 	if(center.y() > aabb[3]) return false;
// 	if(center.z() < aabb[4]) return false;
// 	if(center.z() > aabb[5]) return false;

	for(unsigned int i=0; i<tetrahedron.size(); i++)
	{
		if(tetrahedron[i].tri[0].distancePointPlane(center) > -rad)
			return false;
	}
	return true;
}

bool VolumeLua::sphereIsect(const Sphere& s, int* t) const
{
	const Vector& center = s.pos();
	const double rad = s.radius();
	
	//check for bounding box first
	if(!bb.contains(center, rad)) return false;
// 	if(center.x() + rad < aabb[0]) return false;
// 	if(center.x() - rad > aabb[1]) return false;
// 	if(center.y() + rad < aabb[2]) return false;
// 	if(center.y() - rad > aabb[3]) return false;
// 	if(center.z() + rad < aabb[4]) return false;
// 	if(center.z() - rad > aabb[5]) return false;

	//check if sphere intersects with each tetrahedron

	for(unsigned int i=0; i<tetrahedron.size(); i++)
	{
		if(tetrahedron[i].sphereIsect(s))
		{
			if(t)
			{
				*t = i;
			};
			return true;
		}
	}

	if(t) {*t = 0;};
	return false;
}



#if 0
void VolumeLua::drawWireframe() const
{
	for(int i=0; i<tetrahedron.size(); i++)
	{
		glBegin(GL_LINE_LOOP);
		tetrahedron[i].drawCap();
		glEnd();
	}
}

void VolumeLua::draw() const
{
//	for(int i=0; i<tetrahedron.size(); i++)
//	{
//		glBegin(GL_LINE_LOOP);
//		tetrahedron[i].drawCap();
//		glEnd();
//		tetrahedron[i].drawWireframe();
//	}




	glColor3f(0.6, 0.6, 0.6);
	glBegin(GL_TRIANGLES);

	for(int i=0; i<tetrahedron.size(); i++)
	{
		tetrahedron[i].drawCap();
	}
	glEnd();
}


#endif

double VolumeLua::volume()
{
	double v = 0;
	for(int i=0; i<tetrahedron.size(); i++)
	{
		v += tetrahedron[i].volume();
	}
	return v;
}

void VolumeLua::makeFace(const Vector& center, const vector<Vector>& face)
{
	const int n = face.size();
// 	printf("face size: %i\n", n);
	if(n <= 5)
	{
		for(int i=2; i<n; i++)
		{
//			cout << face[0] << " :: " << face[i-1] << " :: " << face[i] << endl;
			Tetrahedron t(face[0], face[i-1], face[i] , center);
// 			printf("%f\n", t.volume());
			if(t.volume() > 1E-8 && !t.bad)
				tetrahedron.push_back(t);
		}
	}
	else
	{
		Vector m(0,0,0);
		for(int i=0; i<n; i++)
		{
			m += face[i];
		}
		m /= (double)n;

		for(int i=0; i<n; i++)
		{
			Tetrahedron t(face[i], face[(i+1)%n], m , center);

			if(t.volume() > 1E-8 && !t.bad)
				tetrahedron.push_back(t);
		}
	}
}

bool VolumeLua::getFaces(lua_State* L, int index) //const vector<double>& parameters)
{
	bb.reset();
	if(!L)
		return false;
	
	int n = lua_gettop(L) - index + 1;
	
	lua_rawgeti(L, LUA_REGISTRYINDEX, funcref);
	lua_insert(L, index);
	
	if(lua_pcall(L, n, 1, 0))
	{
		cerr << lua_tostring(L, -1) << endl;
		lua_pop(L, 1);
		return false;
	}

	// there is now a list of lists of verts on the stack
	// this is a list of faces.
	center = Vector(0,0,0);
	double numverts = 0;
	double r[3];
	tetrahedron.clear();

	vector< vector<Vector> > faces; //list list verts

	lua_pushnil(L);

	while(lua_next(L, -2))
	{
		vector<Vector> face;
		//key and list of verts on the stack

		lua_pushnil(L);
		while(lua_next(L, -2))
		{
			Vector vr(*lua_tovector(L, -1));
			face.push_back(vr);
			center += vr;
			numverts++;
			lua_pop(L, 1);
		}

		faces.push_back(face);

		lua_pop(L, 1); //pop list
	}

	lua_pop(L, 1); //pop list of list of verts

	if(numverts == 0)
		return true;

	center /= numverts;

// 	if(faces.size())
// 	{
// 		aabb[0] = faces[0][0].x();
// 		aabb[1] = faces[0][0].x();
// 		aabb[2] = faces[0][0].y();
// 		aabb[3] = faces[0][0].y();
// 		aabb[4] = faces[0][0].z();
// 		aabb[5] = faces[0][0].z();
// 	}
	
	for(unsigned int i=0; i<faces.size(); i++)
	{
		makeFace(center, faces[i]);
		
		for(unsigned int j=0; j<faces[i].size(); j++)
		{
			const double x = faces[i][j].x();
			const double y = faces[i][j].y();
			const double z = faces[i][j].z();
			
			bb.include(faces[i][j]);
		}
	}
	return true;
}















int lua_isvolumelua(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "VolumeLua");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

VolumeLua* lua_tovolumelua(lua_State* L, int idx)
{
	VolumeLua** pp = (VolumeLua**)luaL_checkudata(L, idx, "VolumeLua");
	luaL_argcheck(L, pp != NULL, idx, "`VolumeLua' expected");
	return *pp;
}

void lua_pushvolumelua(lua_State* L, VolumeLua* a)
{
	VolumeLua** pp = (VolumeLua**)lua_newuserdata(L, sizeof(VolumeLua**));

	*pp = a;
	luaL_getmetatable(L, "VolumeLua");
	lua_setmetatable(L, -2);
	a->refcount++;
}

static int l_volumelua_new(lua_State* L)
{
	lua_pushvolumelua(L, new VolumeLua());
	return 1;
}

static int l_volumelua_gc(lua_State* L)
{
	VolumeLua* a = lua_tovolumelua(L, 1);
	if(!a) return 0;
	
	a->refcount--;
	if(a->refcount == 0)
		delete a;
	return 0;
}
static int l_volumelua_tostring(lua_State* L)
{
	lua_pushstring(L, "VolumeLua");
	return 1;
}

static int l_volumelua_eq(lua_State* L)
{
	VolumeLua* a = lua_tovolumelua(L, 1);
	if(!a) return 0;

	VolumeLua* b = lua_tovolumelua(L, 2);
	if(!b) return 0;

	lua_pushboolean(L, a==b);
	return 1;
}


static int l_volumelua_setcolor(lua_State* L)
{
	VolumeLua* a = lua_tovolumelua(L, 1);
	if(!a) return 0;

	for(int i=0; i<4; i++)
	{
		if(lua_isnumber(L, 2+i))
		{
			a->color.setComponent(i, lua_tonumber(L, 2+i));
		}
	}
	return 0;
}

static int l_volumelua_getcolor(lua_State* L)
{
	VolumeLua* a = lua_tovolumelua(L, 1);
	if(!a) return 0;

	for(int i=0; i<4; i++)
	{
		lua_pushnumber(L, a->color.component(i));
	}
	return 4;
}

static int l_volumelua_eval(lua_State* L)
{
	VolumeLua* a = lua_tovolumelua(L, 1);
	if(!a) return 0;

// 	vector<double> args;
// 	for(int i=2; i<=lua_gettop(L); i++)
// 	{
// 		args.push_back(lua_tonumber(L, i));
// 	}
	
	a->getFaces(L, 2);
	return 0;
}

static int l_volumelua_setfunction(lua_State* L)
{
	VolumeLua* a = lua_tovolumelua(L, 1);
	if(!a) return 0;

	if(!lua_isfunction(L, -1))
		return luaL_error(L, "VolumeLua.setFunction requires a function");
	
	a->setFunction(L, luaL_ref(L, LUA_REGISTRYINDEX));
	return 0;
}

#include "Atom.h"
static int l_contains(lua_State* L)
{
	VolumeLua* v = lua_tovolumelua(L, 1);
	if(!v) return 0;
	
	if(lua_isatom(L, 2))
	{
		Atom* a = lua_toatom(L, 2);
		
		lua_pushboolean(L, v->contains(a->pos(), a->radius()));
		return 1;
	}

	if(lua_istable(L, 2) || lua_isnumber(L, 2))
	{
		Vector vec;
		
		int r = lua_makevector(L, 2, vec);
		double rad = lua_tonumber(L, 2+r);
		lua_pushboolean(L, v->contains(vec, rad));
		return 1;
	}

	return luaL_error(L, "don't know how to deal with this type");
}

static int l_excludes(lua_State* L)
{
	VolumeLua* v = lua_tovolumelua(L, 1);
	if(!v) return 0;
	
	if(lua_isgroup(L, 2))
	{
		Group* g = lua_togroup(L, 2);
		lua_pushboolean(L, v->excludes(g));
		return 1;
	}
	
	if(lua_isatom(L, 2))
	{
		Atom* a = lua_toatom(L, 2);
		
		lua_pushboolean(L, v->excludes(a->pos(), a->radius()));
		return 1;
	}

	if(lua_istable(L, 2) || lua_isnumber(L, 2) || lua_isvector(L, 2))
	{
		Vector vec;
		
		int r = lua_makevector(L, 2, vec);
		double rad = lua_tonumber(L, 2+r);
		lua_pushboolean(L, v->excludes(vec, rad));
		return 1;
	}

	return luaL_error(L, "don't know how to deal with this type");
}

static int l_volume(lua_State* L)
{
	VolumeLua* v = lua_tovolumelua(L, 1);
	if(!v) return 0;

	lua_pushnumber(L, v->volume());
	return 1;	
}

void lua_registervolumelua(lua_State* L)
{
	static const struct luaL_reg struct_m [] = { //methods
		{"__gc",         l_volumelua_gc},
		{"__eq",         l_volumelua_eq},
		{"__tostring",   l_volumelua_tostring},

		{"setFunction",  l_volumelua_setfunction},
		{"setColor",     l_volumelua_setcolor},
 		{"color",        l_volumelua_getcolor},
		{"eval",         l_volumelua_eval},
		
		{"volume",       l_volume},
		
		{"contains",  l_contains},
		{"excludes",  l_excludes},
		{NULL, NULL}
	};

	luaL_newmetatable(L, "VolumeLua");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
			{"new", l_volumelua_new},
			{NULL, NULL}
	};

	luaL_register(L, "VolumeLua", struct_f);
	lua_pop(L,1);
}

