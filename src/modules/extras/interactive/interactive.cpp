#include <luabaseobject.h>
#include <readline/readline.h>
#include <readline/history.h>


// Let's be honest: the following code is from a different era.
// All examples dealing with readline feel like they come straight
// from 1985. There's bad looking stuff going on here.

static lua_State* _L = 0; // readline is not stateful :(
static int ac_pos = 2; // autocomplete func on stack

static char* history_file = 0;


static void * xmalloc (int size)
{
    void *buf;
 
    buf = malloc (size);
    if (!buf) {
        fprintf (stderr, "Error: Out of memory. Exiting.'n");
        exit (1);
    }
     return buf;
}

static char * dupstr (const char* s) 
{
    char *r;
 
    r = (char*) xmalloc ((strlen (s) + 1));
    strcpy (r, s);
    return (r);
}
 
const char* cmd [] ={ "hello", "world", "hell" ,"word", "quit", " " };


// passed autocomplete responsibility to Lua
char* my_generator(const char* text, int state)
{
    if(_L)
    {
	lua_pushvalue(_L, ac_pos);
	lua_pushstring(_L, text);
	lua_pushinteger(_L, state+1); // for lua niceness
	if(lua_pcall(_L, 2,1,0))
	{
	    fprintf(stderr, "%s\n", lua_tostring(_L, -1));
	    lua_pop(_L, 1);
	    return 0;
	}

	if(lua_isstring(_L, -1))
	{
	    char* g = dupstr(lua_tostring(_L, -1));
	    lua_pop(_L, 1);
	    return g;
	}
	return 0;
    }
    return 0;
}

static char** my_completion( const char * text , int start,  int end)
{
    char **matches;
 
    matches = (char **)NULL;
 
    if (start == 0)
        matches = rl_completion_matches ((char*)text, &my_generator);
    else
        rl_bind_key('\t',rl_abort);
 
    return (matches);
}
 


/* A static variable for holding the line. */
static char *line_read = (char *)NULL;

static char *last_line_read = (char *)NULL;


char* rl_gets (const char* prompt=0)
{
    rl_bind_key('\t',rl_complete);


#if 0
    /* If the buffer has already been allocated, return the memory
       to the free pool. */
    if (line_read)
    {
	free (line_read);
	line_read = (char *)NULL;
    }
#endif

    if (last_line_read)
    {
	free(last_line_read);
	last_line_read = (char *)NULL;
    }

    last_line_read = line_read;

    /* Get a line from the user. */
    line_read = readline (prompt?prompt:"");

    /* If the line has any text in it, save it on the history. */
    if (line_read && *line_read)
    {
        int add_to_history = 0;

        if(last_line_read == 0)
        {
            add_to_history = 1;
        }
        else
        {
            // not strncmp. trusting readline()
            if(strcmp(last_line_read, line_read) != 0)
            {
                add_to_history = 1;
            }
        }



	if(add_to_history)
            add_history (line_read);

        if(history_file) // writing to file every line since many interactive scripts end with ctrl+c
        {
            write_history(history_file);
        }
    }

    return (line_read);
}




static void initialize_readline ()
{
    rl_readline_name = "MagLua_Interactive";
    rl_attempted_completion_function = my_completion;
    rl_completion_entry_function = my_generator; // disable filename auto-complete

    using_history();
    
}

static int l_interactive_setHistoryFile(lua_State* L)
{
    if(history_file)
    {
	free(history_file);
	history_file = 0;
    }

    if(lua_isstring(L, 1))
    {
	const int len = strlen(lua_tostring(L, 1))+1;
	history_file = (char*)malloc(len);
	memcpy(history_file, lua_tostring(L, 1), len);
    }
    

    return 0;
}


static int l_interactive_readline(lua_State* L)
{
    ac_pos = 2;
    _L = L;
    char* line;
    if(lua_isstring(L, 1))
    {
	line = rl_gets(lua_tostring(L, 1));
    }
    else
    {
	line = rl_gets("");
    }

    if(line)
	lua_pushstring(L, line);
    else
	lua_pushnil(L);
    return 1;
}




#include "info.h"
extern "C"
{
int lib_register(lua_State* L);
int lib_version(lua_State* L);
const char* lib_name(lua_State* L);
int lib_main(lua_State* L);
int lib_close(lua_State* L);
}




#include "interactive_luafuncs.h"

int lib_register(lua_State* L)
{
    lua_pushcfunction(L, l_interactive_readline);
    lua_setglobal(L, "_interactive_readline");

    lua_pushcfunction(L, l_interactive_setHistoryFile);
    lua_setglobal(L, "_interactive_setHistoryFile");

    luaL_dofile_interactive_luafuncs(L);

    if(history_file)
	read_history(history_file);


    return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Interactive";
#else
	return "Interactive-Debug";
#endif
}

int lib_main(lua_State* L)
{
    initialize_readline();
    return 0;
}

int lib_close(lua_State* L)
{
    if(history_file)
    {
	write_history(history_file);

        free(history_file);
        history_file = 0;
    }

    return 0;
}
