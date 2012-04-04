#include "QSliderItemLua.h"

QSliderItemLua::QSliderItemLua()
	: QItemLua(hash32("QSliderItemLua"))
{
	changeFunc = 0;
	slider = 0;
}


QSliderItemLua::~QSliderItemLua()
{
	if(changeFunc)
		delete changeFunc;
}

void QSliderItemLua::push(lua_State* L)
{
	luaT_push<QSliderItemLua>(L, this);
}

void QSliderItemLua::setTransparent(float t)
{
	if(!slider) return;

	//slider->setAttribute(Qt::WA_NoSystemBackground, true);
	QPalette pal = slider->palette();
	pal.setBrush(QPalette::Background, QColor(255,255,255,255*(1-t)));
	slider->setPalette(pal);
}



#include "QGraphicsSceneLua.h"
#include <QApplication>
int QSliderItemLua::luaInit(lua_State* L)
{
	QItemLua::luaInit(L);
	if(!scene) return 0;

	//pushbutton = new QPushButton(QApplication::activeWindow());
	slider = new QSlider(0);

	slider->show();

	proxy = scene->addWidget(slider, 0);
	proxy->setPos(0,0);
	proxy->show();

	if(changeFunc)
		delete changeFunc;
	changeFunc = 0;


	if(lua_isfunction(L, 2))
	{
		lua_pushvalue(L, 2);
		changeFunc = new SignalSink(L, luaL_ref(L, LUA_REGISTRYINDEX), slider);
		QObject::disconnect(widget(), 0,0,0);
		//QObject::connect(widget(), SIGNAL(sliderMoved(int)), changeFunc, SLOT(activateInt(int)));
		QObject::connect(widget(), SIGNAL(valueChanged(int)), changeFunc, SLOT(activateInt(int)));
	}

	return 0;
}








static int l_setmovedfunction(lua_State *L)
{
	LUA_PREAMBLE(QSliderItemLua, d, 1);

	if(!lua_isfunction(L, -1))
		return luaL_error(L, "setMovedFunction requires a function");

	if(d->changeFunc)
	{
		delete d->changeFunc;
	}

	d->changeFunc = new SignalSink(L, luaL_ref(L, LUA_REGISTRYINDEX), d->widget());
	QObject::disconnect(d->widget(),0,0,0);
	QObject::connect(d->widget(), SIGNAL(valueChanged(int)), d->changeFunc, SLOT(activateInt(int)));

	return 0;
}

static int l_setv(lua_State *L)
{
	LUA_PREAMBLE(QSliderItemLua, d, 1);
	d->widget()->setOrientation(Qt::Vertical);
	return 0;
}

static int l_seth(lua_State *L)
{
	LUA_PREAMBLE(QSliderItemLua, d, 1);
	d->widget()->setOrientation(Qt::Horizontal);
	return 0;
}

static int l_setrange(lua_State *L)
{
	LUA_PREAMBLE(QSliderItemLua, d, 1);

	if(!d->widget())
		return 0;
	d->widget()->setRange(lua_tointeger(L, 2), lua_tointeger(L, 3));
	return 0;
}

static int l_setvalue(lua_State *L)
{
	LUA_PREAMBLE(QSliderItemLua, d, 1);

	if(!d->widget())
		return 0;
	d->widget()->setValue(lua_tointeger(L, 2));
	return 0;
}
static int l_getvalue(lua_State *L)
{
	LUA_PREAMBLE(QSliderItemLua, d, 1);

	if(!d->widget())
		return 0;
	lua_pushinteger(L, d->widget()->value());
	return 1;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* QSliderItemLua::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, QItemLua::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setMovedFunction", l_setmovedfunction},
		{"setRange", l_setrange},
		{"setVertical", l_setv},
		{"setHorizontal", l_seth},
		{"value", l_getvalue},
		{"setValue", l_setvalue},
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
