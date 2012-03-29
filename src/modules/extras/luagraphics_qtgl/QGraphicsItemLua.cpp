#include "QGraphicsItemLua.h"

QGraphicsItemLua::QGraphicsItemLua(QGraphicsItem* i)
 : LuaBaseObject(hash32("QGraphicsItemLua"))
{
	item = i;
}

QGraphicsItemLua::~QGraphicsItemLua()
{

}


int QGraphicsItemLua::luaInit(lua_State* )
{
	return 0;
}

void QGraphicsItemLua::push(lua_State* L)
{
	luaT_push<QGraphicsItemLua>(L, this);
}

static int l_qgil_getpos(lua_State* L)
{
	LUA_PREAMBLE(QGraphicsItemLua, d, 1);
	if(!d->item) return 0;
	lua_pushnumber(L, d->item->x());
	lua_pushnumber(L, d->item->y());

	return 2;
}

static int l_qgil_setpos(lua_State* L)
{
	LUA_PREAMBLE(QGraphicsItemLua, d, 1);
	if(!d->item) return 0;

	d->item->setPos(lua_tonumber(L, 2), lua_tonumber(L, 3));
	return 0;
}

static int l_qgil_rotate(lua_State* L)
{
	LUA_PREAMBLE(QGraphicsItemLua, d, 1);
	if(!d->item) return 0;

	d->item->rotate(lua_tonumber(L, 2));
	return 0;
}

static int l_qgil_setparent(lua_State* L)
{
	LUA_PREAMBLE(QGraphicsItemLua, d, 1);
	LUA_PREAMBLE(QGraphicsItemLua, p, 2);
	if(!d->item) return 0;

	d->item->setParentItem(p->item);
	return 0;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* QGraphicsItemLua::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"pos",               l_qgil_getpos},
		{"setPos",            l_qgil_setpos},
		{"rotate",            l_qgil_rotate},
		{"setParent",         l_qgil_setparent},
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
