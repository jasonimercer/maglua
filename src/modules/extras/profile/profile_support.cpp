#include "luabaseobject.h"
#include <stdio.h>
#include <string.h>
#include <stack>
#include <string>
#include <map>
#include "profile.h" // for the function references


class call_data_element
{
public:
    call_data_element(string& _name, const void* _id)
        {
            #if 0
            total_inclusive = 0;
            total_exclusive = 0;
            exclusive_start = 0;
            inclusive_start = 0;
            #endif
            recursion_level = 0;
            call_count = 0;
            name = _name;
            id = _id;
        }

    void ensure_size(int r)
        {
            while(total_inclusive.size() <= r) total_inclusive.push_back(0);
            while(total_exclusive.size() <= r) total_exclusive.push_back(0);
            while(inclusive_start.size() <= r) inclusive_start.push_back(0);
            while(exclusive_start.size() <= r) exclusive_start.push_back(0);
        }

    // vectors for recursion levels
    vector<double> total_inclusive;
    vector<double> total_exclusive;
    vector<double> exclusive_start;
    vector<double> inclusive_start; 

    long recursion_level;
    long call_count;

    string name;
    const void* id;

    std::map<const void*, long> functions_I_call;
};


std::stack<const void*> call_stack;
std::map<const void*,call_data_element*> call_data;
std::map<const void*,std::string> function_lookup_list;


int lookup_ref = LUA_REFNIL;
void profile_support_set_lookup(int ref)
{
    lookup_ref = ref;
}



// In the Win32 version we'll store all the time in a single double: seconds
// ...not that there is actually a windows version anymore.
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


void profile_support_init()
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
}

void profile_support_start_timer()
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
}

double profile_support_elapsed()
{
    double seconds = 0;
    double nanoseconds = 0;
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
    seconds = (double)(t1 - t0) / CLOCKS_PER_SEC;
#else
#ifndef MACOSX
    seconds = (t1->tv_sec) - (t0->tv_sec);
    nanoseconds = (t1->tv_nsec) - (t0->tv_nsec);
#else //macosx
    seconds = (t1->tv_sec) - (t0->tv_sec);
    nanoseconds = (t1->tv_usec*1000) - (t0->tv_usec*1000);
#endif
#endif

    return seconds + (nanoseconds)/((double)NANOSEC_PER_SEC);
}



static void l_hook(lua_State* L, lua_Debug* ar)
{
    double elapsed = profile_support_elapsed();
    int event = ar->event;

    // NEED TO THINK HERE 
    // lua_getstack(L, 2, ar);  // get data about stack 2 levels up/down
    lua_getinfo(L, "f", ar); // push function on stack

    lua_pushvalue(L, -1);
    const void* id = lua_topointer(L, -1);
    lua_pop(L, 1);

    if(!id)
        return;

    if(function_lookup_list.find(id) == function_lookup_list.end())
    {
        lua_rawgeti(L, LUA_REGISTRYINDEX, lookup_ref);
        lua_pushvalue(L, -2);
        lua_call(L, 1, 1);

        function_lookup_list.insert( std::pair<const void*,string>(id, string(lua_tostring(L, -1))) );

        lua_pop(L, 1);
    }

    string name = function_lookup_list[id];
    
    if(event == LUA_HOOKCALL)
    {
        if(!call_stack.empty())
        {
            // function running before this new call
            call_data_element* cd = call_data[call_stack.top()];
            if(cd)
            {
                int r = cd->recursion_level;
                cd->ensure_size(r);
                cd->total_exclusive[r] += elapsed - cd->exclusive_start[r];
                cd->functions_I_call[id]++;
            }
        }

        call_stack.push(id);

        if(call_data.find(id) == call_data.end())
            call_data.insert( std::pair<const void*,call_data_element*>(id, new call_data_element(name, id)) );

        call_data_element* new_function = call_data[id];

        new_function->call_count++;
        int r = new_function->recursion_level;

        new_function->ensure_size(r);
        new_function->inclusive_start[r] = elapsed;
        new_function->exclusive_start[r] = elapsed;
        
        new_function->recursion_level++;
    }

    if(event == LUA_HOOKRET || event == LUA_HOOKTAILRET)
    {
        if(!call_stack.empty())
        {
            id = call_stack.top();
            
            call_data_element* cd;
            int r;

            if(call_data.find(id) != call_data.end())
            {
                cd = call_data[id];

                cd->recursion_level--;
                r = cd->recursion_level;
                cd->total_inclusive[r] += (elapsed - cd->inclusive_start[r]);
                cd->total_exclusive[r] += (elapsed - cd->exclusive_start[r]);
            }
            call_stack.pop();

            if(!call_stack.empty())
            {
                id = call_stack.top();
                cd = call_data[id];
                r = cd->recursion_level;
                cd->exclusive_start[r] = elapsed;
            }
        }
    }

    lua_pop(L, 1); // take function off stack
}



void profile_support_start(lua_State* L)
{
    profile_support_init();
    profile_support_start_timer();

    lua_sethook(L, l_hook, LUA_MASKCALL | LUA_MASKRET, 0);
}    

static void id2buf128(const void* id, char* buf128)
{
    snprintf(buf128, 128, "0x%08lX", (long)id);
}


int profile_support_stop(lua_State* L)
{
    double elapsed = profile_support_elapsed();
    char buf128[128];

    lua_newtable(L);
    
    std::map<const void*,call_data_element*>::iterator it;

    for(it=call_data.begin(); it!=call_data.end(); ++it)
    {
        const void* id = it->first;
        call_data_element* cd = it->second;

        if(cd->recursion_level == 0)
        {
            id2buf128(id, buf128);
            
            lua_pushstring(L, buf128);
            lua_newtable(L);
            
            lua_pushstring(L, "name");
            lua_pushstring(L, cd->name.c_str());
            lua_settable(L, -3);
            
            lua_pushstring(L, "inclusive");
            lua_pushnumber(L, cd->total_inclusive[0]);
            lua_settable(L, -3);
            
            lua_pushstring(L, "exclusive");
            lua_pushnumber(L, cd->total_exclusive[0]);
            lua_settable(L, -3);
            
            lua_pushstring(L, "call_count");
            lua_pushinteger(L, cd->call_count);
            lua_settable(L, -3);
            
            lua_pushstring(L, "recursion_level");
            lua_pushinteger(L, cd->recursion_level);
            lua_settable(L, -3);
            
            
            
            lua_pushstring(L, "calls");
            lua_newtable(L);
            std::map<const void*, long>::iterator it2;
            for(it2=cd->functions_I_call.begin(); it2!=cd->functions_I_call.end(); ++it2)
            {
                id2buf128(it2->first, buf128);
                lua_pushstring(L, buf128);
                lua_pushinteger(L, it2->second);
                lua_settable(L, -3);
            }
            lua_settable(L, -3);
            
            
            lua_settable(L, -3);
        }
    }


    
    lua_pushnumber(L, elapsed);
    return 2;
}
