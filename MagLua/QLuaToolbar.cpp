#include "QLuaToolbar.h"
#include "MainWindow.h"
#include "QLuaAction.h"


QLuaToolBar::QLuaToolBar(lua_State* L, QToolBar *tb) :
	QLuaWidget(L, tb)
{

}

QLuaToolBar::~QLuaToolBar()
{

}


int lua_istoolbar(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "ToolBar");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaToolBar* lua_toluatoolbar(lua_State* L, int idx)
{
	QLuaToolBar** pp = (QLuaToolBar**)luaL_checkudata(L, idx, "ToolBar");
	luaL_argcheck(L, pp != NULL, idx, "`ToolBar' expected");
	return *pp;
}

QToolBar* lua_totoolbar(lua_State* L, int idx)
{
	QLuaToolBar* pp = lua_toluatoolbar(L, idx);
	if(pp)
		return (QToolBar*)(pp->widget);
	return 0;
}

void lua_pushluatoolbar(lua_State* L, QLuaToolBar* c)
{
	QLuaToolBar** pp = (QLuaToolBar**)lua_newuserdata(L, sizeof(QLuaToolBar**));

	*pp = c;
	luaL_getmetatable(L, "ToolBar");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_toolbar_new(lua_State* L)
{
	QString txt = lua_tostring(L, 1);
	lua_pushluatoolbar(L, new QLuaToolBar(L, Singleton.mainWindow->addToolBar(txt)));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaToolBar* c = lua_toluatoolbar(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_istoolbar(L, 1))
	{
		lua_pushstring(L, "ToolBar");
		return 1;
	}
	return 0;
}

static int l_settext(lua_State* L)
{
	QToolBar* c = lua_totoolbar(L, 1);
	if(!c) return 0;

	c->setWindowTitle(lua_tostring(L, 2));

	return 0;
}
static int l_gettext(lua_State* L)
{
	QToolBar* c = lua_totoolbar(L, 1);
	if(!c) return 0;

	lua_pushstring(L, c->windowTitle().toStdString().c_str());

	return 1;
}

static int l_addaction(lua_State* L)
{
	QToolBar* c = lua_totoolbar(L, 1);
	if(!c) return 0;

	QLuaAction* a = lua_toluaaction(L, 2);
	if(!a) return 0;

	c->addAction(a->action);
	return 0;
}

static int l_addseparator(lua_State* L)
{
	QToolBar* c = lua_totoolbar(L, 1);
	if(!c) return 0;

	c->addSeparator();

	return 0;
}





void lua_registertoolbar(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"windowTitle",      l_gettext},
	  {"setWindowTitle",   l_settext},
	  {"addAction",        l_addaction},
	  {"addSeparator",        l_addseparator},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "ToolBar");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_toolbar_new},
		{NULL, NULL}
	};

	luaL_register(L, "ToolBar", struct_f);
	lua_pop(L,1);
}
