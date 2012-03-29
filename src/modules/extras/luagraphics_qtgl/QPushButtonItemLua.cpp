#include "QPushButtonItemLua.h"

QPushButtonItemLua::QPushButtonItemLua()
	: LuaBaseObject(hash32("QPushButtonItemLua"))
{

	pressFunc = 0;
	pushbutton = 0;
	proxy = 0;
}


QPushButtonItemLua::~QPushButtonItemLua()
{
	if(pressFunc)
		delete pressFunc;

	//	if(hasPressedFunction)
	//	{
	//		luaL_unref(L, LUA_REGISTRYINDEX, pressedFunction);
	//	}
	//	if(highlighter)
	//		delete highlighter;
	//	cout << "Deleting QTextEdit" << endl;
}

#include "QGraphicsSceneLua.h"
#include <QApplication>
int QPushButtonItemLua::luaInit(lua_State* L)
{
	if(luaT_is<QGraphicsSceneLua>(L, 1))
	{
		QGraphicsSceneLua* s = luaT_to<QGraphicsSceneLua>(L, 1);
		luaT_inc<QGraphicsSceneLua>(s);
		pushbutton = new QPushButton(QApplication::activeWindow());
		pushbutton->show();

		proxy = s->scene->addWidget(pushbutton, 0);
		proxy->setPos(0,0);
		proxy->show();

		if(lua_isfunction(L, 2))
		{
			lua_pushvalue(L, 2);
			pressFunc = new SignalSink(L, luaL_ref(L, LUA_REGISTRYINDEX), pushbutton);
			QObject::connect(d->widget(), SIGNAL(pressed()), d->pressFunc, SLOT(activate()));
		}
	}
	return 0;
}








static int l_pbil_setpressedfunction(lua_State *L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);

	if(!lua_isfunction(L, -1))
		return luaL_error(L, "setPressedFunction requires a function");

	if(d->pressFunc)
	{
		delete d->pressFunc;
	}

	d->pressFunc = new SignalSink(L, luaL_ref(L, LUA_REGISTRYINDEX), pushbutton);
	QObject::connect(d->widget(), SIGNAL(pressed()), d->pressFunc, SLOT(activate()));

	return 0;
}

static int l_pbil_settext(lua_State *L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);


	d->widget()->setText(lua_tostring(L, 2));
	return 0;
}

static int l_pbil_gettext(lua_State *L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);

	lua_pushstring(L, d->widget()->text().toStdString().c_str());
	return 1;
}

#if 0
#include "QGraphicsItemLua.h"
static int l_pbil_item(lua_State* L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);
	luaT_push<QGraphicsItemLua>(L, new QGraphicsItemLua(d->item()));
	return 1;
}
#endif


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* QPushButtonItemLua::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setText",           l_pbil_settext},
		{"text",              l_pbil_gettext},
		{"setPressedFunction", l_pbil_setpressedfunction},
		//{"item",              l_pbil_item},
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
