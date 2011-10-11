#include "QLuaMenu.h"
#include "Classes.h"

#include "QLuaAction.h"

int lua_ismenu(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Menu");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaMenu* lua_toluamenu(lua_State* L, int idx)
{
	if(!lua_ismenu(L, idx))
		return 0;
	QLuaMenu** pp = (QLuaMenu**)luaL_checkudata(L, idx, "Menu");
	luaL_argcheck(L, pp != NULL, idx, "`Menu' expected");
	return *pp;
}

QMenu* lua_tomenu(lua_State* L, int idx)
{
	QLuaMenu* a = lua_toluamenu(L, idx);
	if(!a) return 0;

	return (QMenu*)a->widget;
}

void lua_pushluamenu(lua_State* L, QLuaMenu* c)
{
	QLuaMenu** pp = (QLuaMenu**)lua_newuserdata(L, sizeof(QLuaMenu**));

	*pp = c;
	luaL_getmetatable(L, "Menu");
	lua_setmetatable(L, -2);
	c->refcount++;
}

void lua_pushmenu(lua_State* L, QMenu* s)
{
	lua_pushluamenu(L, new QLuaMenu(L, s));
}

static int l_menu_new(lua_State* L)
{
	QString txt = lua_tostring(L, 1);
	lua_pushmenu(L, new QMenu(txt));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaMenu* c = lua_toluamenu(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_ismenu(L, 1))
	{
		lua_pushstring(L, "Menu");
		return 1;
	}
	return 0;
}

static int l_addseparator(lua_State* L)
{
	QMenu* c = lua_tomenu(L, 1);
	if(!c) return 0;

	c->addSeparator();
	return 0;
}

static int l_clear(lua_State* L)
{
	QMenu* c = lua_tomenu(L, 1);
	if(!c) return 0;

	c->clear();
	return 0;
}

static int l_additem(lua_State* L)
{
	QLuaMenu* cc = lua_toluamenu(L, 1);
	QMenu* c = lua_tomenu(L, 1);
	if(!c) return 0;

	QString text = lua_tostring(L, 2);
	QString icon = lua_tostring(L, 3);

	QAction* a = 0;

	if(text.length())
	{
		if(icon.length())
		{
			a = c->addAction(QIcon(icon), text);
		}
		else
		{
			a = c->addAction(text);
		}
	}

	if(a)
	{
		lua_pushaction(L, a);
		cc->addChild(lua_toluaaction(L, -1));
		return 1;
	}

	return 0;
}

void lua_registermenu(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"addItem",    l_additem},
	  {"addSeparator",    l_addseparator},
	  {"clear",      l_clear},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Menu");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_menu_new},
		{NULL, NULL}
	};

	luaL_register(L, "Menu", struct_f);
	lua_pop(L,1);
}


