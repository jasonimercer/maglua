#include "SignalSink.h"

SignalSink::SignalSink(lua_State* _L, int funcref, QObject *parent) :
    QObject(parent)
{
	L = _L;
	ref = funcref;
}

SignalSink::~SignalSink()
{
	if(L)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, ref);
	}
}


void SignalSink::activate()
{
	if(!L) return;
	if(ref == LUA_REFNIL) return;

	lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
	lua_call(L, 0, 0);
//	lua_gc(L, LUA_GCCOLLECT, 0);
}
