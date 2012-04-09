#include "QPushButtonItemLua.h"

QPushButtonItemLua::QPushButtonItemLua()
	: QItemLua(hash32(lineage(0)))
{
	pressFunc = 0;
	pushbutton = 0;
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

void QPushButtonItemLua::push(lua_State* L)
{
	luaT_push<QPushButtonItemLua>(L, this);
}


void QPushButtonItemLua::setTransparent(float t)
{
	if(!pushbutton) return;

	//slider->setAttribute(Qt::WA_NoSystemBackground, true);
	QPalette pal = pushbutton->palette();
	pal.setBrush(QPalette::Background, QColor(255,255,255,255*(1-t)));
	pushbutton->setPalette(pal);
}



#include "QGraphicsSceneLua.h"
#include <QApplication>
int QPushButtonItemLua::luaInit(lua_State* L)
{
	QItemLua::luaInit(L);
	if(!scene) return 0;

	//pushbutton = new QPushButton(QApplication::activeWindow());
	pushbutton = new QPushButton(0);
	pushbutton->show();
	setTransparent();

	proxy = scene->addWidget(pushbutton, 0);
	proxy->setPos(0,0);
	proxy->show();

	if(pressFunc)
		delete pressFunc;
	pressFunc = 0;

	for(int i=2; i<=3; i++)
	{
		if(lua_isfunction(L, i) && !pressFunc)
		{
			lua_pushvalue(L, i);
			pressFunc = new SignalSink(L, luaL_ref(L, LUA_REGISTRYINDEX), pushbutton);
			QObject::connect(widget(), SIGNAL(pressed()), pressFunc, SLOT(activate()));
		}
		else
		{
			if(lua_isstring(L, i))
			{
				pushbutton->setText(lua_tostring(L, i));
			}
		}
	}
	return 0;
}








static int l_setpressedfunction(lua_State *L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);

	if(!lua_isfunction(L, -1))
		return luaL_error(L, "setPressedFunction requires a function");

	if(d->pressFunc)
	{
		delete d->pressFunc;
	}

	d->pressFunc = new SignalSink(L, luaL_ref(L, LUA_REGISTRYINDEX), d->widget());
	QObject::connect(d->widget(), SIGNAL(pressed()), d->pressFunc, SLOT(activate()));

	return 0;
}

static int l_setrepeat(lua_State *L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);

	d->widget()->setAutoRepeat(lua_toboolean(L, 2));
	return 0;
}

static int l_settext(lua_State *L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);
	d->widget()->setText(lua_tostring(L, 2));
	return 0;
}

static int l_gettext(lua_State *L)
{
	LUA_PREAMBLE(QPushButtonItemLua, d, 1);

	lua_pushstring(L, d->widget()->text().toStdString().c_str());
	return 1;
}

const luaL_Reg* QPushButtonItemLua::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)return m;

	merge_luaL_Reg(m, QItemLua::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setRepeat",           l_setrepeat},
		{"setText",             l_settext},
		{"text",                l_gettext},
		{"setPressedFunction",  l_setpressedfunction},
		{NULL,NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
