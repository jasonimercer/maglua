/* Jason Mercer's Custom Timer Routines. Compile with -lm */
#include "mtimer.h"

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "info.h"

// in the Win32 version we'll store all the time in a single double: seconds
#ifdef WIN32
 #include <windows.h>
#endif 

#ifdef CLOCK_THREAD_CPU_TIME
#define CLOCKTYPE CLOCK_THREAD_CPU_TIME
#else
#define CLOCKTYPE CLOCK_REALTIME
#endif

#ifndef NANOSEC_PER_SEC
#  define NANOSEC_PER_SEC 1000000000
#endif


Timer::Timer()
	: Encodable(hash32("Timer"))
{
#ifdef WIN32
	//not using structures
#else
#ifndef MACOSX
	t0 = (struct timespec*)malloc(sizeof(struct timespec));
	t1 = (struct timespec*)malloc(sizeof(struct timespec));
#else
	t0 = (struct timeval*)malloc(sizeof(struct timeval));
	t1 = (struct timeval*)malloc(sizeof(struct timeval));
#endif
#endif
	seconds = 0;
	nanoseconds = 0;
	paused = 1;
	refcount = 0;
}


Timer::~Timer()
{
#ifndef WIN32
	free(t0);
	free(t1);
#endif
}


void Timer::encode(buffer* b)
{
	int running = !paused;
	
	if(running)
		stop();

	double t = get_time();

	if(running)
		start();
	
	encodeInteger(running, b);
	encodeDouble(t, b);
}

int Timer::decode(buffer* b)
{
	stop();
	reset();
	
	int running = decodeInteger(b);
	double t = decodeDouble(b);
	
	set_time(t);
	if(running)
		start();
}
	

void Timer::fixTime()
{
#ifndef WIN32
	while(nanoseconds < 0)
	{
		nanoseconds+=NANOSEC_PER_SEC;
		seconds--;
	}
	
	while(nanoseconds > NANOSEC_PER_SEC)
	{
		nanoseconds-=NANOSEC_PER_SEC;
		seconds++;
	}
#endif
}

long Timer::get_seconds()
{
	fixTime();
#ifdef WIN32
	return (long) floor(seconds);
#else
	return seconds;
#endif
}

long Timer::get_nanoseconds()
{
	fixTime();
	return nanoseconds;
}

double Timer::get_time()
{
	fixTime();
#ifdef WIN32
	return seconds;
#else
	return (double)(seconds) + (double)(nanoseconds)/((double)NANOSEC_PER_SEC);
#endif
}

void Timer::set_time(double t)
{
#ifdef WIN32
	seconds = t;
	nanoseconds = 0;
#else
	seconds = floor(t);
	nanoseconds = (t - floor(t)) * NANOSEC_PER_SEC;
#endif
}


void Timer::reset()
{
	seconds = 0;
	nanoseconds = 0;
	paused = 1;
}

void Timer::start()
{
#ifdef WIN32
	t0 = clock();
#else
#ifndef MACOSX
	clock_gettime(CLOCKTYPE, (struct timespec*)t0);
#else
	gettimeofday((struct timeval *)t0, NULL);
#endif
#endif
	paused = 0;  
}

void Timer::stop()
{
#ifdef WIN32
	t1 = clock();
#else
#ifndef MACOSX
	clock_gettime(CLOCKTYPE, (struct timespec*)t1);
#else
	gettimeofday( (struct timeval *)t1, NULL);
#endif
#endif

#ifdef WIN32
	seconds += (double)(t1 - t0) / CLOCKS_PER_SEC;
#else
#ifndef MACOSX
	seconds += (t1->tv_sec) - (t0->tv_sec);
	nanoseconds += (t1->tv_nsec) - (t0->tv_nsec);
#else //macosx
	seconds += (t1->tv_sec) - (t0->tv_sec);
	nanoseconds += (t1->tv_usec*1000) - (t0->tv_usec*1000);
#endif
#endif
	paused = 1;

	fixTime();
}

void Timer::pause()
{
	if(paused)
		start();
	else
		stop();
}









int lua_istimer(lua_State* L, int idx)
{
	if(!lua_isuserdata(L, idx))
			return 0;
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "MERCER.timer");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

Timer* lua_totimer(lua_State* L, int idx)
{
	if(!lua_istimer(L, idx))
		return 0;
	
	Timer** pp = (Timer**)luaL_checkudata(L, idx, "MERCER.timer");
	if(!pp)
	{
		luaL_error(L, "not a Timer");
		return 0;
	}
	return *pp;
}

void lua_pushtimer(lua_State* L, Encodable* _t)
{
	Timer* t = dynamic_cast<Timer*>(_t);
	if(!t) return;
	t->refcount++;
	
	Timer** pp = (Timer**)lua_newuserdata(L, sizeof(Timer**));
	
	*pp = t;
	luaL_getmetatable(L, "MERCER.timer");
	lua_setmetatable(L, -2);
}

static int l_new(lua_State* L)
{
	lua_pushtimer(L, new Timer());
	return 1;
}

static int l_gc(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;

	timer->refcount--;
	if(timer->refcount == 0)
		delete timer;
	
	return 0;
}

static int l_tostring(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	lua_pushstring(L, "Timer");
	return 1;
}

static int l_reset(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	timer->reset();
	return 0;
}


static int l_start(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	timer->start();
	return 0;
}

static int l_pause(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	timer->pause();
	
	return 0;
}

static int l_stop(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	timer->stop();
	
	return 0;
}

static int l_getsec(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	lua_pushinteger(L, timer->get_seconds());
	return 1;
}
static int l_getnano(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	lua_pushinteger(L, timer->get_nanoseconds());
	return 1;
}
static int l_get(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	lua_pushnumber(L, timer->get_time());
	return 1;
}

static int l_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.timer");
	return 1;
}


static int l_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Create a timer to profile code");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_new)
	{
		lua_pushstring(L, "Create a new Timer Operator.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Timer object");
		return 3;
	}	
	if(func == l_start)
	{
		lua_pushstring(L, "Reset and start timer.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}	
	
	if(func == l_stop)
	{
		lua_pushstring(L, "Stop timer.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}	
	if(func == l_pause)
	{
		lua_pushstring(L, "Pause timer");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}	
	if(func == l_getsec)
	{
		lua_pushstring(L, "Get elapsed seconds since last reset/start");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 number: elapsed seconds");
		return 3;
	}	
	if(func == l_getnano)
	{
		lua_pushstring(L, "Get elapsed nanoseconds since last reset/start");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 number: elapsed nanoseconds");
		return 3;
	}
	
	if(func == l_get)
	{
		lua_pushstring(L, "Get elapsed seconds since last reset/start");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 number: elapsed seconds");
		return 3;
	}

	return 0;
}

static Encodable* newThing()
{
	return new Timer;
}

void registerTimer(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",       l_gc},
		{"__tostring", l_tostring},
		{"start",      l_start},
		{"stop",       l_stop},
		{"pause",      l_pause},
		{"reset",      l_reset},
		{"seconds",    l_getsec},
		{"nanoseconds",l_getnano},
		{"elapsed",       l_get},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.timer");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_new},
		{"help",                l_help},
		{"metatable",           l_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Timer", functions);
	lua_pop(L,1);	

	Factory_registerItem(hash32("Timer"), newThing, lua_pushtimer, "Timer");
}


extern "C"
{
TIMER_API int lib_register(lua_State* L);
TIMER_API int lib_deps(lua_State* L);
TIMER_API int lib_version(lua_State* L);
TIMER_API const char* lib_name(lua_State* L);
TIMER_API int lib_main(lua_State* L);
}

TIMER_API int lib_register(lua_State* L)
{
	registerTimer(L);
	return 0;
}

TIMER_API int lib_deps(lua_State* L)
{
	return 0;
}

TIMER_API int lib_version(lua_State* L)
{
	return __revi;
}

TIMER_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Timer";
#else
	return "Timer-Debug";
#endif
}

TIMER_API int lib_main(lua_State* L)
{
	return 0;
}



