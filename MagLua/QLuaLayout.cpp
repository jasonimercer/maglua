#include "QLuaLayout.h"
#include "Classes.h"

int lua_islayout(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Layout");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaLayout* lua_tolayout(lua_State* L, int idx)
{
	if(!lua_islayout(L, idx))
		return 0;
	QLuaLayout** pp = (QLuaLayout**)luaL_checkudata(L, idx, "Layout");
	luaL_argcheck(L, pp != NULL, idx, "`Layout' expected");
	return *pp;
}

void lua_pushlayout(lua_State* L, QLuaLayout* c)
{
	QLuaLayout** pp = (QLuaLayout**)lua_newuserdata(L, sizeof(QLuaLayout**));

	*pp = c;
	luaL_getmetatable(L, "Layout");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_layout_new(lua_State* L)
{
	lua_pushlayout(L, new QLuaLayout(L, new QGridLayout));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaLayout* c = lua_tolayout(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_islayout(L, 1))
	{
		lua_pushstring(L, "Layout");
		return 1;
	}
	return 0;
}


static int l_add(lua_State* L)
{
	QLuaLayout* cc = lua_tolayout(L, 1);
	if(!cc) return 0;

	QGridLayout* layout = (QGridLayout*)cc->layout;
	int r = lua_tointeger(L, 3) - 1;
	int c = lua_tointeger(L, 4) - 1;

	if(lua_isluawidget(L, 2))
	{
		QLuaLayout* l2 = lua_tolayout(L, 2);
		if(l2)
		{
			cc->addChild(l2);
			QWidget* ww = new QWidget;
			layout->addWidget(ww, r, c);
			ww->setLayout((QGridLayout*)l2->layout);
		}
		else
		{
			cc->addChild(lua_toluawidget(L, 2));

			layout->addWidget(lua_towidget(L, 2), r, c);
		}
	}
	else
	{
		if(lua_isstring(L, 2))
		{
			QString o = lua_tostring(L, 2);

			o = o.at(0);
			int h = o.compare("H", Qt::CaseInsensitive) == 0;
			int v = o.compare("V", Qt::CaseInsensitive) == 0;

			if(h | v)
			{
				QWidget* w = new QWidget;
				w->setLayout(new QVBoxLayout);

				if(v)
					w->layout()->addItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
				else
					w->layout()->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));

				layout->addWidget(w, r, c);
			}
		}
	}

	return 0;
}

void lua_registerlayout(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"add",        l_add},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Layout");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_layout_new},
		{NULL, NULL}
	};

	luaL_register(L, "Layout", struct_f);
	lua_pop(L,1);
}


