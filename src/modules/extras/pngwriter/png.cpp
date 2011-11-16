#include "png.h"
#include <pngwriter.h>

//  PNGWRITER_API

typedef struct Lpng
{
    pngwriter* w;
    int refcount;
} Lpng;


Lpng* checkPNG(lua_State* L, int idx)
{
    Lpng** pp = (Lpng**)luaL_checkudata(L, idx, "MERCER.png");
    luaL_argcheck(L, pp != NULL, 1, "`PNGWriter' expected");
    return *pp;
}

void lua_pushPNG(lua_State* L, Lpng* p)
{
    p->refcount++;
    Lpng** pp = (Lpng**)lua_newuserdata(L, sizeof(Lpng**));
        
    *pp = p;
    luaL_getmetatable(L, "MERCER.png");
    lua_setmetatable(L, -2);
}


static int l_new(lua_State* L)
{
    const char* filename = lua_tostring(L, 1);
    int w, h;

    if(!filename)
	return luaL_error(L, "PNGWriter.new requires a filename, width and height");

    if(!lua_isnumber(L, 2) | !lua_isnumber(L, 3))
	return luaL_error(L, "PNGWriter.new requires a filename, width and height");

    w = lua_tointeger(L, 2);
    h = lua_tointeger(L, 3);

    Lpng* pp = new Lpng;
    pp->w = new pngwriter(w, h, 0, filename);
    pp->refcount = 0;

    lua_pushPNG(L, pp);

    return 1;
}

static int l_tostring(lua_State* L)
{
    lua_pushstring(L, "PNGWriter");
    return 1;	
}


static int l_gc(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

    p->refcount--;

    if(p->refcount <= 0)
    {
	if(p->w)
	{
	    p->w->close();
	    delete p->w;
	}
	delete p;
    }
	
    return 0;
}


static int l_plot(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

    int x = lua_tointeger(L, 2);
    int y = lua_tointeger(L, 3);

    double r = lua_tonumber(L, 4);
    double g = lua_tonumber(L, 5);
    double b = lua_tonumber(L, 6);

    if(r > 1) r = 1;
    if(g > 1) g = 1;
    if(b > 1) b = 1;

    if(r < 0) r = 0;
    if(g < 0) g = 0;
    if(b < 0) b = 0;

    p->w->plot(x, y, r, g, b);
    return 0;
}


static int l_plot_text(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

	if(lua_gettop(L) != 10)
	{
		return luaL_error(L, "plot_text expects (font.ttf, fontsize, x, y, angle, text, red, green, blue)\n");
	}
	
	const char* face_path = lua_tostring(L, 2);
	int fontsize = lua_tointeger(L, 3);
	
	char* fp = new char[strlen(face_path)+1];
	strcpy(fp, face_path);
	
	int startx = lua_tointeger(L, 4);
	int starty = lua_tointeger(L, 5);
	
	double angle = lua_tonumber(L, 6);
	const char* text = lua_tostring(L, 7);
	char* txt = new char[strlen(text)+1];
	strcpy(txt, text);
	
    double r = lua_tonumber(L, 8);
    double g = lua_tonumber(L, 9);
    double b = lua_tonumber(L,10);

    if(r > 1) r = 1;
    if(g > 1) g = 1;
    if(b > 1) b = 1;

    if(r < 0) r = 0;
    if(g < 0) g = 0;
    if(b < 0) b = 0;

    p->w->plot_text(fp, fontsize, startx, starty, angle, txt, r, g, b);
	
	delete [] fp;
	delete [] txt;
    return 0;
}

// out.plot(rr, rrr, 65535, 0, 0);

int lib_registerpngwriter(lua_State* L)
{
	static const struct luaL_reg methods [] = {
		{"__gc",         l_gc},
		{"__tostring",   l_tostring},
		{"plot",         l_plot},
		{"plot_text",    l_plot_text},
		
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.png");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);
	lua_settable(L, -3);
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_new},
		{NULL, NULL}
	};
		
	luaL_register(L, "PNGWriter", functions);
	lua_pop(L,1);
	return 0;
}

#include "info.h"
extern "C"
{
PNGWRITER_API int lib_register(lua_State* L);
PNGWRITER_API int lib_version(lua_State* L);
PNGWRITER_API const char* lib_name(lua_State* L);
PNGWRITER_API int lib_main(lua_State* L, int argc, char** argv);
}

PNGWRITER_API int lib_register(lua_State* L)
{
	lib_registerpngwriter(L);
	return 0;
}

PNGWRITER_API int lib_version(lua_State* L)
{
	return __revi;
}

PNGWRITER_API const char* lib_name(lua_State* L)
{
	return "PNGWriter";
}

PNGWRITER_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}
