#include "QItemLua.h"
#include "QGraphicsSceneLua.h"
#include <QWidget>

QItemLua::QItemLua(int etype)
	: LuaBaseObject(etype)
{
	proxy = 0;
	scene = 0;
	scene_lua = 0;
}

QItemLua::~QItemLua()
{
	luaT_dec<QGraphicsSceneLua>(scene_lua);
}

int QItemLua::luaInit(lua_State* L)
{
	QGraphicsSceneLua* s = luaT_to<QGraphicsSceneLua>(L, 1);
	luaT_inc<QGraphicsSceneLua>(s);
	luaT_dec<QGraphicsSceneLua>(scene_lua);
	scene = s->scene;
	scene_lua = s;
	return 0;
}

void QItemLua::push(lua_State* L)
{
	luaT_push<QItemLua>(L, this);
}

void QItemLua::setGeometry(int x, int y, int w, int h)
{
	if(!proxy) return;
	proxy->widget()->setGeometry(x,y,w,h);
}

QRect	QItemLua::geometry ()
{
	if(!proxy) return QRect();
	return proxy->widget()->geometry();
}

void QItemLua::setTransparent(float t)
{
	return;
}




static int l_setgeom(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);
	i->setGeometry(
				lua_tointeger(L, 2),
				lua_tointeger(L, 3),
				lua_tointeger(L, 4),
				lua_tointeger(L, 5));
	return 0;
}

static int l_move(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);

	QRect r = i->geometry();

	i->setGeometry(
				lua_tointeger(L, 2),
				lua_tointeger(L, 3),
				r.width(),
				r.height());
	return 0;
}

static int l_resize(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);

	QRect r = i->geometry();

	i->setGeometry(
				r.x(),
				r.y(),
				lua_tointeger(L, 2),
				lua_tointeger(L, 3));
	return 0;
}

static int l_getgeom(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);

	QRect r = i->geometry();

	lua_pushinteger(L, r.x());
	lua_pushinteger(L, r.y());
	lua_pushinteger(L, r.width());
	lua_pushinteger(L, r.height());
	return 4;
}

static int l_getwidth(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);
	lua_pushinteger(L, i->geometry().width());
	return 1;
}
static int l_getheight(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);
	lua_pushinteger(L, i->geometry().height());
	return 1;
}
static int l_setwidth(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);
	QRect r = i->geometry();

	i->setGeometry(
				r.x(),
				r.y(),
				lua_tointeger(L, 2),
				r.height());
	return 0;
}
static int l_setheight(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, i, 1);
	QRect r = i->geometry();

	i->setGeometry(
				r.x(),
				r.y(),
				r.width(),
				lua_tointeger(L, 2));
	return 0;
}
static int l_settbg(lua_State* L)
{
	LUA_PREAMBLE(QItemLua, d, 1);

	if(lua_isboolean(L, 2))
	{
		if(lua_toboolean(L, 2))
		{
			d->setTransparent(1.0);
		}
		else
		{
			d->setTransparent(0.0);
		}
	}
	else
	{
		d->setTransparent(lua_tonumber(L, 2));
	}
	return 0;

}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* QItemLua::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"move", l_move},
		{"resize", l_resize},
		{"width", l_getwidth},
		{"height", l_getheight},
		{"setWidth", l_setwidth},
		{"setHeight", l_setheight},
		{"setGeometry", l_setgeom},
		{"geometry", l_getgeom},
		{"setTransparent", l_settbg},
		{NULL,NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
