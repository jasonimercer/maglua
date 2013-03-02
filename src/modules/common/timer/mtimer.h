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
	
	void fixTime();
	long get_seconds();
	long get_nanoseconds();
	double get_time();
	void set_time(double t);
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


#ifdef WIN32
	double seconds;
#else
	long seconds;
#endif
	long nanoseconds;
	int paused;
};
#endif
