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
	: LuaBaseObject(hash32("Timer"))
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
}


Timer::~Timer()
{
#ifndef WIN32
	free(t0);
	free(t1);
#endif
}

int Timer::luaInit(lua_State* L)
{
	LuaBaseObject::luaInit(L);
	reset();
	return 0;
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
	return 0;
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
	reset();
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





static int l_reset(lua_State* L)
{
	LUA_PREAMBLE(Timer, timer, 1);	
	timer->reset();
	return 0;
}


static int l_start(lua_State* L)
{
	LUA_PREAMBLE(Timer, timer, 1);	
	timer->start();
	return 0;
}

static int l_pause(lua_State* L)
{
	LUA_PREAMBLE(Timer, timer, 1);	
	timer->pause();
	return 0;
}

static int l_stop(lua_State* L)
{
	LUA_PREAMBLE(Timer, timer, 1);	
	timer->stop();
	return 0;
}

static int l_getsec(lua_State* L)
{
	LUA_PREAMBLE(Timer, timer, 1);	
	lua_pushinteger(L, timer->get_seconds());
	return 1;
}
static int l_getnano(lua_State* L)
{
	LUA_PREAMBLE(Timer, timer, 1);	
	lua_pushinteger(L, timer->get_nanoseconds());
	return 1;
}
static int l_get(lua_State* L)
{
	LUA_PREAMBLE(Timer, timer, 1);	
	lua_pushnumber(L, timer->get_time());
	return 1;
}

int Timer::help(lua_State* L)
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

	
	lua_CFunction func = lua_tocfunction(L, 1);
	

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


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Timer::luaMethods()
{
	if(m[127].name)return m;

	static const luaL_Reg _m[] =
	{
		{"start",      l_start},
		{"stop",       l_stop},
		{"pause",      l_pause},
		{"reset",      l_reset},
		{"seconds",    l_getsec},
		{"nanoseconds",l_getnano},
		{"elapsed",       l_get},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
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
	luaT_register<Timer>(L);
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



