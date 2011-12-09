extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#ifdef WIN32
 #ifdef TIMER_EXPORTS
  #define TIMER_API __declspec(dllexport)
 #else
  #define TIMER_API __declspec(dllimport)
 #endif
#else
 #define TIMER_API 
#endif



TIMER_API typedef struct Timer Timer;

TIMER_API void lua_pushtimer(lua_State* L, Timer* t);
TIMER_API int lua_istimer(lua_State* L, int idx);
TIMER_API Timer* lua_totimer(lua_State* L, int idx);
TIMER_API void registertimer(lua_State* L);

TIMER_API struct Timer* new_timer();
TIMER_API void free_timer(struct Timer* t);
TIMER_API void reset_timer(struct Timer* t);
TIMER_API void start_timer(struct Timer* t);
TIMER_API void stop_timer(struct Timer* t);
TIMER_API double get_time(struct Timer* t);
