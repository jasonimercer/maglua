#include "QLuaTimer.h"
#include "MainWindow.h"


QLuaTimer::QLuaTimer(lua_State* _L) :
	QObject(Singleton.mainWindow)
{
	L = _L;
	timer = new QTimer(this);
	funcref = LUA_NOREF;
	refcount = 0;
	gcd = 0;
	connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));
}

QLuaTimer::~QLuaTimer()
{
	if(funcref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, funcref);
	}
	delete timer;
}

void QLuaTimer::setFunction(int ref)
{
	if(funcref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, funcref);
	}
	funcref = ref;
}

void QLuaTimer::oneshot(int ms)
{
	timer->setSingleShot(true);
	timer->start(ms);
}

void QLuaTimer::start(int ms)
{
	timer->setSingleShot(false);
	timer->start(ms);
}

void QLuaTimer::stop()
{
	timer->stop();
}

void QLuaTimer::timeout()
{
	if(funcref != LUA_NOREF)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, funcref);

		if(lua_pcall(L, 0, 0, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
			lua_pop(L, 1);
		}
		lua_gc(L, LUA_GCCOLLECT, 0);
	}

	if(!timer->isActive())
	{
		if(gcd)
			deleteLater();
	}

}

bool QLuaTimer::running()
{
	return timer->isActive();
}





int lua_istimer(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Timer");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaTimer* lua_toluatimer(lua_State* L, int idx)
{
	QLuaTimer** pp = (QLuaTimer**)luaL_checkudata(L, idx, "Timer");
	luaL_argcheck(L, pp != NULL, idx, "`Timer' expected");
	return *pp;
}

QTimer* lua_totimer(lua_State* L, int idx)
{
	QLuaTimer* pp = lua_toluatimer(L, idx);
	if(pp)
		return pp->timer;
	return 0;
}

void lua_pushluatimer(lua_State* L, QLuaTimer* c)
{
	QLuaTimer** pp = (QLuaTimer**)lua_newuserdata(L, sizeof(QLuaTimer**));

	*pp = c;
	luaL_getmetatable(L, "Timer");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_timer_new(lua_State* L)
{
	lua_pushluatimer(L, new QLuaTimer(L));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaTimer* c = lua_toluatimer(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
	{
		if(c->running())
		{
			c->gcd = 1; //timer will delete itself when it fires
		}
		else
		{
			delete c;
		}
	}
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_istimer(L, 1))
	{
		lua_pushstring(L, "Timer");
		return 1;
	}
	return 0;
}


static int l_setfunc(lua_State* L)
{
	QLuaTimer* c = lua_toluatimer(L, 1);
	if(!c) return 0;

	c->setFunction(luaL_ref(L, LUA_REGISTRYINDEX));
	return 0;
}

static int l_start(lua_State* L)
{
	QLuaTimer* c = lua_toluatimer(L, 1);
	if(!c) return 0;

	c->start(lua_tointeger(L, 2));
	return 0;
}

static int l_oneshot(lua_State* L)
{
	QLuaTimer* c = lua_toluatimer(L, 1);
	if(!c) return 0;

	c->oneshot(lua_tointeger(L, 2));
	return 0;
}

static int l_stop(lua_State* L)
{
	QLuaTimer* c = lua_toluatimer(L, 1);
	if(!c) return 0;

	c->stop();
	return 0;
}

static int l_running(lua_State* L)
{
	QLuaTimer* c = lua_toluatimer(L, 1);
	if(!c) return 0;

	lua_pushboolean(L, c->running());

	return 1;
}

void lua_registertimer(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"setFunction", l_setfunc},
	  {"start",       l_start},
	  {"oneShot",     l_oneshot},
	  {"stop",        l_stop},
	  {"running",     l_running},

	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Timer");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_timer_new},
		{NULL, NULL}
	};

	luaL_register(L, "Timer", struct_f);
	lua_pop(L,1);
}
