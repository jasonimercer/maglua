/* Jason Mercer's Custom Timer Routines. Compile with -lm */
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "info.h"
#include "mtimer.h"

// in the Win32 version we'll store all the time in a single double: seconds
#ifdef WIN32
 #include <windows.h>
#endif 

#ifndef MTIMER_STRUCT
#define MTIMER_STRUCT
typedef struct Timer
{
#ifdef WIN32
	clock_t  t0;
	clock_t  t1;
#else
#ifndef MACOSX
	struct timespec* t0; /* start time */
	struct timespec* t1; /* end time */
#else
	struct timeval* t0; /* gettimeofday data */
	struct timeval* t1;
#endif
#endif

#ifdef WIN32
	double seconds;
#else
	long seconds;
#endif
	long nanoseconds;
	int paused;
	int refcount;
} Timer;
#endif

#ifdef CLOCK_THREAD_CPU_TIME
#define CLOCKTYPE CLOCK_THREAD_CPU_TIME
#else
#define CLOCKTYPE CLOCK_REALTIME
#endif

#ifndef NANOSEC_PER_SEC
#  define NANOSEC_PER_SEC 1000000000
#endif
struct Timer* new_timer()
{
	Timer* t = (Timer*)malloc(sizeof(Timer));

#ifdef WIN32
	//not using structures
#else
#ifndef MACOSX
	t->t0 = (struct timespec*)malloc(sizeof(struct timespec));
	t->t1 = (struct timespec*)malloc(sizeof(struct timespec));
#else
	t->t0 = (struct timeval*)malloc(sizeof(struct timeval));
	t->t1 = (struct timeval*)malloc(sizeof(struct timeval));
#endif
#endif
	t->seconds = 0;
	t->nanoseconds = 0;
	t->paused = 1;
	t->refcount = 0;
	return t;
}


void free_timer(struct Timer* t)
{
#ifndef WIN32
	free(t->t0);
	free(t->t1);
#endif
	free(t);
}

static void fixTime(struct Timer* t)
{
#ifndef WIN32
	while(t->nanoseconds < 0)
	{
		t->nanoseconds+=NANOSEC_PER_SEC;
		t->seconds--;
	}
	
	while(t->nanoseconds > NANOSEC_PER_SEC)
	{
		t->nanoseconds-=NANOSEC_PER_SEC;
		t->seconds++;
	}
#endif
}

static long get_seconds(struct Timer* t)
{
	fixTime(t);
#ifdef WIN32
	return (long) floor(t->seconds);
#else
	return t->seconds;
#endif
}

static long get_nanoseconds(struct Timer* t)
{
	fixTime(t);
	return t->nanoseconds;
}

double get_time(struct Timer* t)
{
	fixTime(t);
#ifdef WIN32
	return t->seconds;
#else
	return (double)(t->seconds) + (double)(t->nanoseconds)/((double)NANOSEC_PER_SEC);
#endif
}

void reset_timer(struct Timer* t)
{
	t->seconds = 0;
	t->nanoseconds = 0;
	t->paused = 1;
}

void start_timer(struct Timer* t)
{
#ifdef WIN32
	t->t0 = clock();
#else
#ifndef MACOSX
	clock_gettime(CLOCKTYPE, (struct timespec*)t->t0);
#else
	gettimeofday((struct timeval *)t->t0, NULL);
#endif
#endif
	t->paused = 0;  
}

void stop_timer(struct Timer* t)
{
#ifdef WIN32
	t->t1 = clock();
#else
#ifndef MACOSX
	clock_gettime(CLOCKTYPE, (struct timespec*)t->t1);
#else
	gettimeofday( (struct timeval *)t->t1, NULL);
#endif
#endif

#ifdef WIN32
	t->seconds += (double)(t->t1 - t->t0) / CLOCKS_PER_SEC;
#else
#ifndef MACOSX
	t->seconds += (t->t1->tv_sec) - (t->t0->tv_sec);
	t->nanoseconds += (t->t1->tv_nsec) - (t->t0->tv_nsec);
#else //macosx
	t->seconds += (t->t1->tv_sec) - (t->t0->tv_sec);
	t->nanoseconds += (t->t1->tv_usec*1000) - (t->t0->tv_usec*1000);
#endif
#endif
	t->paused = 1;

	fixTime(t);
}

static void pause_timer(struct Timer* t)
{
	if(t->paused)
		start_timer(t);
	else
		stop_timer(t);
}









TIMER_API int lua_istimer(lua_State* L, int idx)
{
	if(!lua_isuserdata(L, idx))
			return 0;
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "MERCER.timer");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

TIMER_API Timer* lua_totimer(lua_State* L, int idx)
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

TIMER_API void lua_pushtimer(lua_State* L, Timer* timer)
{
	timer->refcount++;
	Timer** pp = (Timer**)lua_newuserdata(L, sizeof(Timer**));
	
	*pp = timer;
	luaL_getmetatable(L, "MERCER.timer");
	lua_setmetatable(L, -2);
}

static int l_new(lua_State* L)
{
	lua_pushtimer(L, new_timer());
	return 1;
}

static int l_gc(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;

	timer->refcount--;
	if(timer->refcount == 0)
		free_timer(timer);
	
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
	
	reset_timer(timer);
	return 0;
}


static int l_start(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	start_timer(timer);
	return 0;
}

static int l_pause(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	pause_timer(timer);
	return 0;
}

static int l_stop(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	stop_timer(timer);
	return 0;
}

static int l_getsec(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	lua_pushinteger(L, get_seconds(timer));
	return 1;
}
static int l_getnano(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	lua_pushinteger(L, get_nanoseconds(timer));
	return 1;
}
static int l_get(lua_State* L)
{
	Timer* timer = lua_totimer(L, 1);
	if(!timer) return 0;
	
	lua_pushnumber(L, get_time(timer));
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
}


extern "C"
{
TIMER_API int lib_register(lua_State* L);
TIMER_API int lib_deps(lua_State* L);
TIMER_API int lib_version(lua_State* L);
TIMER_API const char* lib_name(lua_State* L);
TIMER_API int lib_main(lua_State* L, int argc, char** argv);
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
	return "Timer";
}

TIMER_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}



