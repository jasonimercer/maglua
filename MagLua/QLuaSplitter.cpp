#include "QLuaSplitter.h"
#include "Classes.h"


int lua_issplitter(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Splitter");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaSplitter* lua_tosplitter(lua_State* L, int idx)
{
	QLuaSplitter** pp = (QLuaSplitter**)luaL_checkudata(L, idx, "Splitter");
	luaL_argcheck(L, pp != NULL, idx, "`Splitter' expected");
	return *pp;
}

void lua_pushsplitter(lua_State* L, QLuaSplitter* c)
{
	QLuaSplitter** pp = (QLuaSplitter**)lua_newuserdata(L, sizeof(QLuaSplitter**));

	*pp = c;
	luaL_getmetatable(L, "Splitter");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_setorientation(lua_State* L);
static int l_splitter_new(lua_State* L)
{
	lua_pushsplitter(L, new QLuaSplitter(L, new QSplitter));

	if(lua_isstring(L, 1))
	{
		lua_pushcfunction(L, l_setorientation);
		lua_pushvalue(L, -2);
		lua_pushvalue(L, 1);
		lua_call(L, 2, 0);
	}

	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaSplitter* c = lua_tosplitter(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_issplitter(L, 1))
	{
		lua_pushstring(L, "Splitter");
		return 1;
	}
	return 0;
}


static int l_add(lua_State* L)
{
	QLuaSplitter* c = lua_tosplitter(L, 1);
	if(!c) return 0;

	QLuaWidget* ww = lua_toluawidget(L, 2);
	if(ww)
	{
		c->addChild(ww);
		((QSplitter*)c->widget)->addWidget(ww->widget);
	}


	return 0;
}

static int l_getorientation(lua_State* L)
{
	QLuaSplitter* c = lua_tosplitter(L, 1);
	if(!c) return 0;

	if(((QSplitter*)c->widget)->orientation() ==  Qt::Horizontal)
	{
		lua_pushstring(L, "Horizontal");
	}
	else
	{
		lua_pushstring(L, "Vertical");
	}
	return 1;
}

static int l_setorientation(lua_State* L)
{
	QLuaSplitter* c = lua_tosplitter(L, 1);
	if(!c) return 0;

	QString o = lua_tostring(L, 2);

	o = o.at(0);

	if(o.compare("H", Qt::CaseInsensitive) == 0)
	{
		((QSplitter*)c->widget)->setOrientation(Qt::Horizontal);
	}
	if(o.compare("V", Qt::CaseInsensitive) == 0)
	{
		((QSplitter*)c->widget)->setOrientation(Qt::Vertical);
	}
	return 0;
}




void lua_registersplitter(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"add",        l_add},
	  {"orientation", l_getorientation},
	  {"setOrientation", l_setorientation},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Splitter");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_splitter_new},
		{NULL, NULL}
	};

	luaL_register(L, "Splitter", struct_f);
	lua_pop(L,1);
}


