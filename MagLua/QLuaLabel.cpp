#include "QLuaLabel.h"
#include "Classes.h"

int lua_islabel(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Label");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaLabel* lua_tolualabel(lua_State* L, int idx)
{
	if(!lua_islabel(L, idx))
		return 0;
	QLuaLabel** pp = (QLuaLabel**)luaL_checkudata(L, idx, "Label");
	luaL_argcheck(L, pp != NULL, idx, "`Label' expected");
	return *pp;
}
QLabel* lua_tolabel(lua_State* L, int idx)
{
	QLuaLabel* ll = lua_tolualabel(L, idx);
	if(!ll) return 0;
	return (QLabel*)ll->widget;
}

void lua_pushlabel(lua_State* L, QLuaLabel* c)
{
	QLuaLabel** pp = (QLuaLabel**)lua_newuserdata(L, sizeof(QLuaLabel**));

	*pp = c;
	luaL_getmetatable(L, "Label");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_label_new(lua_State* L)
{
	QString txt = lua_tostring(L, 1);
	lua_pushlabel(L, new QLuaLabel(L, new QLabel(txt)));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaLabel* c = lua_tolualabel(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_islabel(L, 1))
	{
		lua_pushstring(L, "Label");
		return 1;
	}
	return 0;
}

static int l_settext(lua_State* L)
{
	QLabel* c = lua_tolabel(L, 1);
	if(!c) return 0;

	c->setText(lua_tostring(L, 2));

	return 0;
}
static int l_gettext(lua_State* L)
{
	QLabel* c = lua_tolabel(L, 1);
	if(!c) return 0;

	lua_pushstring(L, c->text().toStdString().c_str());

	return 1;
}

static int l_setwordwrap(lua_State* L)
{
	QLabel* c = lua_tolabel(L, 1);
	if(!c) return 0;

	c->setWordWrap(lua_toboolean(L, 2));

	return 0;
}

void lua_registerlabel(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"text",       l_gettext},
	  {"setText",    l_settext},
	  {"setWordWrap",    l_setwordwrap},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Label");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_label_new},
		{NULL, NULL}
	};

	luaL_register(L, "Label", struct_f);
	lua_pop(L,1);
}


