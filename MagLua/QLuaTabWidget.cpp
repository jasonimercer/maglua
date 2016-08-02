#include "QLuaTabWidget.h"
#include "QLuaLayout.h"
#include "Classes.h"


int QLuaTabWidget::addTab(QLuaWidget* lw, QString title)
{
	addChild(lw);

	int i = ((QTabWidget*)widget)->addTab(lw->widget, title);
	return i;
}


int lua_istabwidget(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "TabWidget");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaTabWidget* lua_totabwidget(lua_State* L, int idx)
{
	QLuaTabWidget** pp = (QLuaTabWidget**)luaL_checkudata(L, idx, "TabWidget");
	luaL_argcheck(L, pp != NULL, idx, "`TabWidget' expected");
	return *pp;
}

void lua_pushtabwidget(lua_State* L, QLuaTabWidget* c)
{
	QLuaTabWidget** pp = (QLuaTabWidget**)lua_newuserdata(L, sizeof(QLuaTabWidget**));

	*pp = c;
	luaL_getmetatable(L, "TabWidget");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_tabwidget_new(lua_State* L)
{
	lua_pushtabwidget(L, new QLuaTabWidget(L, new QTabWidget));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaTabWidget* c = lua_totabwidget(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_istabwidget(L, 1))
	{
		lua_pushstring(L, "TabWidget");
		return 1;
	}
	return 0;
}


static int l_add(lua_State* L)
{
	QLuaTabWidget* c = lua_totabwidget(L, 1);
	if(!c) return 0;

//	c->widget->addWidget(lua_towidget(L, 2));

	return 0;
}

static int l_setcurrentindex(lua_State* L)
{
	QLuaTabWidget* c = lua_totabwidget(L, 1);
	if(!c) return 0;

	((QTabWidget*)c->widget)->setCurrentIndex(lua_tointeger(L, -1)-1);

	return 0;
}

static int l_addtab(lua_State* L)
{
	QLuaTabWidget* c = lua_totabwidget(L, 1);
	if(!c) return 0;

	QString title = lua_tostring(L, 2);
	int i;
	QLuaWidget* w = lua_toluawidget(L, 3);
	if(w)
	{
		i = c->addTab(w, title);
	}


	bool l = lua_islayout(L, 3);
	if(l)
	{
		QWidget* w = new QWidget();
		i = ((QTabWidget*)c->widget)->addTab(w, title);

		if(w->layout())
			delete w->layout();

		w->setLayout(lua_tolayout(L, 3)->layout);
	}


	if(!w && !l)
	{
		i = ((QTabWidget*)c->widget)->addTab(new QWidget(), title);
	}

	lua_pushinteger(L, i+1);
//	c->widget->addWidget(lua_towidget(L, 2));

	return 1;
}

void lua_registertabwidget(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"addTab",     l_addtab},
	  {"add",        l_add},
	  {"setCurrentIndex", l_setcurrentindex},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "TabWidget");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_tabwidget_new},
		{NULL, NULL}
	};

	luaL_register(L, "TabWidget", struct_f);
	lua_pop(L,1);
}


