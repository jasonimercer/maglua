#include "QGraphicsSceneLua.h"

QGraphicsSceneLua::QGraphicsSceneLua(QGraphicsScene* s)
	: LuaBaseObject(hash32("QGraphicsSceneLua"))
{
	scene = s;
}

QGraphicsSceneLua::~QGraphicsSceneLua()
{

}

int QGraphicsSceneLua::luaInit(lua_State* L)
{
	if(lua_isuserdata(L, 1))
	{
		//yuck
		scene = (QGraphicsScene*)lua_touserdata(L, 1);
	}
	return 0;
}

void QGraphicsSceneLua::push(lua_State* L)
{
	luaT_push<QGraphicsSceneLua>(L, this);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* QGraphicsSceneLua::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{NULL,NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
