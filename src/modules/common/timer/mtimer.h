#ifndef MTIMER_DEF
#define MTIMER_DEF

#ifdef WIN32
 #ifdef TIMER_EXPORTS
  #define TIMER_API __declspec(dllexport)
 #else
  #define TIMER_API __declspec(dllimport)
 #endif
#else
 #define TIMER_API 
#endif

#include "luabaseobject.h"

class TIMER_API Timer : public LuaBaseObject
{
public:
	Timer();
	virtual ~Timer();
	
	LINEAGE1("Timer")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	double get_time();
	void set_time(double t);
        void accumulate();
	void reset();
	void start();
	void stop();
	void pause();


	void encode(buffer* b);
	int decode(buffer* b);
	
	
	
#ifdef WIN32
	long t0;
	long t1;
#else
#ifndef MACOSX
	struct timespec* t0; /* start time */
	struct timespec* t1; /* end time */
#else
	struct timeval* t0; /* gettimeofday data */
	struct timeval* t1;
#endif
#endif


	double seconds;
	long nanoseconds;
	int paused;
};
#endif
