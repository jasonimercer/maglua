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

static int l_plot_blend(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

    int x = lua_tointeger(L, 2);
    int y = lua_tointeger(L, 3);
	double opacity = lua_tonumber(L, 4);
	
    double r = lua_tonumber(L, 5);
    double g = lua_tonumber(L, 6);
    double b = lua_tonumber(L, 7);

    if(r > 1) r = 1;
    if(g > 1) g = 1;
    if(b > 1) b = 1;

    if(r < 0) r = 0;
    if(g < 0) g = 0;
    if(b < 0) b = 0;

    p->w->plot_blend(x, y, opacity, r, g, b);
    return 0;
}

static int l_pixelat(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

	if(lua_gettop(L) != 3)
	{
		return luaL_error(L, "pixelAt expects (x,y)\n");
	}
	
	int x = lua_tointeger(L, 2);
	int y = lua_tointeger(L, 3);
	
	lua_pushnumber(L, p->w->dread(x,y,1));
	lua_pushnumber(L, p->w->dread(x,y,2));
	lua_pushnumber(L, p->w->dread(x,y,3));
	return 3;
}


static int l_line(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;
	
	if(lua_gettop(L) != 8)
	{
		return luaL_error(L, "line expects (x1, y1, x2, y2, red, green, blue)\n");
	}

	int x1 = lua_tointeger(L, 2);
	int y1 = lua_tointeger(L, 3);
	int x2 = lua_tointeger(L, 4);
	int y2 = lua_tointeger(L, 5);
	
	double r = lua_tonumber(L, 6);
	double g = lua_tonumber(L, 7);
	double b = lua_tonumber(L, 8);
	
	p->w->line(x1, y1, x2, y2, r, g, b);
	return 0;

}


static int l_line_blend(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;
	
	if(lua_gettop(L) != 9)
	{
		return luaL_error(L, "line_blend expects (x1, y1, x2, y2, opacity, red, green, blue)\n");
	}

	int x1 = lua_tointeger(L, 2);
	int y1 = lua_tointeger(L, 3);
	int x2 = lua_tointeger(L, 4);
	int y2 = lua_tointeger(L, 5);
	
	double opacity = lua_tonumber(L, 6);
	
	double r = lua_tonumber(L, 7);
	double g = lua_tonumber(L, 8);
	double b = lua_tonumber(L, 9);
	
	p->w->line_blend(x1, y1, x2, y2, opacity, r, g, b);
	return 0;
}


static int l_arrow(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;
	
	if(lua_gettop(L) != 10)
	{
		return luaL_error(L, "arrow expects (x1, y1, x2, y2, size, head_angle(rad), red, green, blue)\n");
	}

	int x1 = lua_tointeger(L, 2);
	int y1 = lua_tointeger(L, 3);
	int x2 = lua_tointeger(L, 4);
	int y2 = lua_tointeger(L, 5);
	
	int size = lua_tointeger(L, 6);
	double head_angle = lua_tointeger(L, 7);
	
	double r = lua_tonumber(L, 8);
	double g = lua_tonumber(L, 9);
	double b = lua_tonumber(L,10);
	
	p->w->arrow(x1, x2, y1, y2, size, head_angle, r, g, b);
	return 0;
}

static int l_filledarrow(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;
	
	if(lua_gettop(L) != 10)
	{
		return luaL_error(L, "filledarrow expects (x1, y1, x2, y2, size, head_angle(rad), red, green, blue)\n");
	}

	int x1 = lua_tointeger(L, 2);
	int y1 = lua_tointeger(L, 3);
	int x2 = lua_tointeger(L, 4);
	int y2 = lua_tointeger(L, 5);
	
	int size = lua_tointeger(L, 6);
	double head_angle = lua_tointeger(L, 7);
	
	double r = lua_tonumber(L, 8);
	double g = lua_tonumber(L, 9);
	double b = lua_tonumber(L,10);
	
	p->w->filledarrow(x1, x2, y1, y2, size, head_angle, r, g, b);
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


static int l_plot_text_utf8(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

	if(lua_gettop(L) != 10)
	{
		return luaL_error(L, "plot_text_utf8 expects (font.ttf, fontsize, x, y, angle, text, red, green, blue)\n");
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

    p->w->plot_text_utf8(fp, fontsize, startx, starty, angle, txt, r, g, b);
	
	delete [] fp;
	delete [] txt;
    return 0;
}


static int l_get_text_width(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

	if(lua_gettop(L) != 4)
	{
		return luaL_error(L, "get_text_width expects (font.ttf, fontsize, text)\n");
	}
	
	const char* face_path = lua_tostring(L, 2);
	int fontsize = lua_tointeger(L, 3);
	const char* text = lua_tostring(L, 4);
	
	lua_pushinteger(L, p->w->get_text_width((char*)face_path, fontsize, (char*)text));
	return 1;
}

error("array needs to be moved to script so we're not linking against cpu/gpu implementation")
#include "../../cpu/array/array.h"
static int l_drawarray(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

	LUA_PREAMBLE(dArray, a, 2);
	
	double scale = lua_tonumber(L, 3);
	
	double r = lua_tonumber(L, 4);
	double g = lua_tonumber(L, 5);
	double b = lua_tonumber(L, 6);
	

	int imin, imax;
	a->min(imin);
	a->max(imax);
	
	double min = a->data()[imin];
	double max = a->data()[imax];
	
	double idiff = max - min;
	if(idiff > 0)
		idiff = 1.0/idiff;
	
	for(int x=0; x<a->nx; x++)
	{
		for(int y=0; y<a->ny; y++)
		{
			double v = (a->get(x,y) - min) * idiff;
			
			p->w->filledsquare(x*scale, y*scale, (x+1)*scale, (y+1)*scale, r*v, g*v, b*v);
		}
	}
	
	
	return 0;
}

static int l_filledsquare(lua_State* L)
{
    Lpng* p = checkPNG(L, 1);
    if(!p) return 0;

	if(lua_gettop(L) != 8)
	{
		return luaL_error(L, "filledsquare expects (x1, y1, x2, y2, r, g, b)\n");
	}
	
	const int x1 = lua_tointeger(L, 2);
	const int y1 = lua_tointeger(L, 3);
	const int x2 = lua_tointeger(L, 4);
	const int y2 = lua_tointeger(L, 5);

	const double r = lua_tonumber(L, 6);
	const double g = lua_tonumber(L, 7);
	const double b = lua_tonumber(L, 8);

	p->w->filledsquare(x1, y1, x2, y2, r, g, b);
	return 0;
}

// out.plot(rr, rrr, 65535, 0, 0);

int lib_registerpngwriter(lua_State* L)
{
	static const struct luaL_reg methods [] = {
		{"__gc",         l_gc},
		{"__tostring",   l_tostring},
		{"plot",         l_plot},
		{"plot_blend",   l_plot_blend},
		{"plot_text",    l_plot_text},
		{"plot_text_utf8",    l_plot_text_utf8},
		{"read",         l_pixelat},
		{"line",         l_line},
		{"line_blend",   l_line_blend},
		{"arrow",        l_arrow},
		{"filledarrow",  l_filledarrow},
		{"get_text_width", l_get_text_width},
		{"filledsquare",  l_filledsquare},
		{"drawArray", l_drawarray},
		
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
PNGWRITER_API int lib_main(lua_State* L);
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

PNGWRITER_API int lib_main(lua_State* L)
{
	return 0;
}
