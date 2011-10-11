#include "QLuaPushButton.h"
#include "Classes.h"

#include <iostream>
#include <QErrorMessage>
#include "MainWindow.h"
using namespace std;

QLuaPushButton::QLuaPushButton(lua_State *L, QPushButton *w) : QLuaWidget(L, w)
{
	funcref = LUA_NOREF;
	connect(w, SIGNAL(pressed()), this, SLOT(pressed()));
}

QLuaPushButton::~QLuaPushButton()
{
	if(funcref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, funcref);
	}
}


void QLuaPushButton::pressed()
{
	if(funcref != LUA_NOREF)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, funcref);

		if(lua_pcall(L, 0, 0, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
			QErrorMessage* msg = new QErrorMessage(Singleton.mainWindow);
			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
			lua_pop(L, lua_gettop(L));
		}
		lua_gc(L, LUA_GCCOLLECT, 0);
	}
}

void QLuaPushButton::setPressedFunction(int ref)
{
	if(funcref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, funcref);
	}
	funcref = ref;
}



int lua_ispushbutton(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "PushButton");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaPushButton* lua_topushbutton(lua_State* L, int idx)
{
	QLuaPushButton** pp = (QLuaPushButton**)luaL_checkudata(L, idx, "PushButton");
	luaL_argcheck(L, pp != NULL, idx, "`PushButton' expected");
	return *pp;
}

void lua_pushpushbutton(lua_State* L, QLuaPushButton* c)
{
	QLuaPushButton** pp = (QLuaPushButton**)lua_newuserdata(L, sizeof(QLuaPushButton**));

	*pp = c;
	luaL_getmetatable(L, "PushButton");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_pushbutton_new(lua_State* L)
{
	QString txt = lua_tostring(L, 1);
	lua_pushpushbutton(L, new QLuaPushButton(L, new QPushButton(txt)));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaPushButton* c = lua_topushbutton(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_ispushbutton(L, 1))
	{
		lua_pushstring(L, "PushButton");
		return 1;
	}
	return 0;
}

static int l_settext(lua_State* L)
{
	QLuaPushButton* c = lua_topushbutton(L, 1);
	if(!c) return 0;

	((QPushButton*)c->widget)->setText(lua_tostring(L, 2));

	return 0;
}
static int l_gettext(lua_State* L)
{
	QLuaPushButton* c = lua_topushbutton(L, 1);
	if(!c) return 0;

	lua_pushstring(L, ((QPushButton*)c->widget)->text().toStdString().c_str());

	return 1;
}

static int l_setfunc(lua_State* L)
{
	QLuaPushButton* c = lua_topushbutton(L, 1);
	if(!c) return 0;

	c->setPressedFunction(luaL_ref(L, LUA_REGISTRYINDEX));
	return 0;
}
static int l_getfunc(lua_State* L)
{
	QLuaPushButton* c = lua_topushbutton(L, 1);
	if(!c) return 0;

	lua_rawgeti(L, LUA_REGISTRYINDEX, c->funcref);

	return 1;
}

void lua_registerpushbutton(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"text",       l_gettext},
	  {"setText",    l_settext},
	  {"function",   l_getfunc},
	  {"setFunction",l_setfunc},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "PushButton");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_pushbutton_new},
		{NULL, NULL}
	};

	luaL_register(L, "PushButton", struct_f);
	lua_pop(L,1);
}


