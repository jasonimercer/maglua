#include "QLuaLineEdit.h"
#include "Classes.h"

int lua_islualineedit(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "LineEdit");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaLineEdit* lua_tolualineedit(lua_State* L, int idx)
{
	if(!lua_islualineedit(L, idx))
		return 0;

	QLuaLineEdit** pp = (QLuaLineEdit**)luaL_checkudata(L, idx, "LineEdit");
	luaL_argcheck(L, pp != NULL, idx, "`LineEdit' expected");
	return *pp;
}

void lua_pushlualineedit(lua_State* L, QLuaLineEdit* c)
{
	QLuaLineEdit** pp = (QLuaLineEdit**)lua_newuserdata(L, sizeof(QLuaLineEdit**));

	*pp = c;
	luaL_getmetatable(L, "LineEdit");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_lineedit_new(lua_State* L)
{
	QString txt = lua_tostring(L, 1);
	lua_pushlualineedit(L, new QLuaLineEdit(L, new QLineEdit(txt)));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaLineEdit* c = lua_tolualineedit(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_islualineedit(L, 1))
	{
		lua_pushstring(L, "LineEdit");
		return 1;
	}
	return 0;
}

static int l_settext(lua_State* L)
{
	QLuaLineEdit* c = lua_tolualineedit(L, 1);
	if(!c) return 0;

	((QLineEdit*)c->widget)->setText(lua_tostring(L, 2));

	return 0;
}
static int l_gettext(lua_State* L)
{
	QLuaLineEdit* c = lua_tolualineedit(L, 1);
	if(!c) return 0;

	lua_pushstring(L, ((QLineEdit*)c->widget)->text().toStdString().c_str());

	return 1;
}

void lua_registerlineedit(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"text",       l_gettext},
	  {"setText",    l_settext},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "LineEdit");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_lineedit_new},
		{NULL, NULL}
	};

	luaL_register(L, "LineEdit", struct_f);
	lua_pop(L,1);
}
