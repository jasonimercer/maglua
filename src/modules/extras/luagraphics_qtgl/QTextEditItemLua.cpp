#include "QTextEditItemLua.h"
#include <QErrorMessage>
#include <iostream>
#include <QApplication>

#include "QGraphicsSceneLua.h"
using namespace std;

QTextEditItemLua::QTextEditItemLua()
	: LuaBaseObject(hash32("QTextEditItemLua"))
{

	textedit = 0;
	proxy = 0;
	highlighter = 0;

	setTransparentBackgound(1);
}


QTextEditItemLua::~QTextEditItemLua()
{
	//	if(hasPressedFunction)
	//	{
	//		luaL_unref(L, LUA_REGISTRYINDEX, pressedFunction);
	//	}
	//	if(highlighter)
	//		delete highlighter;
	//	cout << "Deleting QTextEdit" << endl;
}


int QTextEditItemLua::luaInit(lua_State* L)
{
	if(luaT_is<QGraphicsSceneLua>(L, 1))
	{
		QGraphicsSceneLua* s = luaT_to<QGraphicsSceneLua>(L, 1);
		luaT_inc<QGraphicsSceneLua>(s);
		textedit = new QTextEdit(QApplication::activeWindow());
		textedit->show();
		textedit->setTabStopWidth(40);

		proxy = s->scene->addWidget(textedit, 0);
		proxy->setPos(0,0);
		proxy->show();

		setTransparentBackgound(1);
	}
	return 0;
}

void QTextEditItemLua::push(lua_State* L)
{
	luaT_push<QTextEditItemLua>(L, this);
}


void QTextEditItemLua::setTransparentBackgound(float t)
{
	if(!textedit) return;
	textedit->setAttribute(Qt::WA_NoSystemBackground, true);
	QPalette pal = textedit->palette();
	pal.setBrush(QPalette::Base, QColor(255,255,255,255*(1-t)));
	textedit->setPalette(pal);
}

#if 0
void QTextEditItemLua::pressed()
{
	//	if(hasPressedFunction)
	//	{
	//		lua_rawgeti(L, LUA_REGISTRYINDEX, pressedFunction);

	//		if(lua_pcall(L, 0, 0, 0))
	//		{
	//			cerr << lua_tostring(L, -1) << endl;
	//			QErrorMessage* msg = new QErrorMessage(Singleton.mainWindow);
	//			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
	//			lua_pop(L, lua_gettop(L));
	//		}
	//		lua_gc(L, LUA_GCCOLLECT, 0);
	//	}
}
#endif
















static int l_settext(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	d->widget()->setPlainText(lua_tostring(L, 2));
	return 0;
}

static int l_sethtml(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	d->widget()->setHtml(lua_tostring(L, 2));
	return 0;
}

static int l_gettext(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	lua_pushstring(L, d->widget()->toPlainText().toStdString().c_str());
	return 1;
}

static int l_addhtml(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	d->widget()->moveCursor(QTextCursor::End);
	d->widget()->insertHtml(lua_tostring(L, 2));
	return 0;
}

static int l_addtext(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	d->widget()->moveCursor(QTextCursor::End);
	d->widget()->insertPlainText(lua_tostring(L, 2));
	return 0;
}

static int l_setframe(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	d->widget()->setFrameStyle(lua_tointeger(L, 2));
	return 0;
}

static int l_setsize(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->item()) return 0;

	int x = lua_tointeger(L, 2);
	int y = lua_tointeger(L, 3);

	d->item()->setMinimumWidth(x);
	d->item()->setMaximumWidth(x);

	d->item()->setMinimumHeight(y);
	d->item()->setMaximumHeight(y);
	return 0;
}

static int l_setwidth(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->item()) return 0;

	int v = lua_tointeger(L, 2);
	d->item()->setMinimumWidth(v);
	d->item()->setMaximumWidth(v);
	return 0;
}
static int l_setheight(lua_State *L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->item()) return 0;

	int v = lua_tointeger(L, 2);
	d->item()->setMinimumHeight(v);
	d->item()->setMaximumHeight(v);
	return 0;
}

static int l_setreadonly(lua_State* L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	bool b = lua_toboolean(L, 2);

	if(b)
	{
		d->widget()->setTextInteractionFlags(Qt::NoTextInteraction);
	}
	else
	{
		d->widget()->setTextInteractionFlags(Qt::TextEditorInteraction);
	}


	//	d->widget()->setReadOnly(b);
	//	d->widget()->setReadOnly();

	return 0;
}

#include "QGraphicsItemLua.h"
//static int l_item(lua_State* L)
//{
//	LUA_PREAMBLE(QTextEditItemLua, d, 1);
//	luaT_push<QGraphicsItemLua>(L, new QGraphicsItemLua(d->item()));
//	return 1;
//}

static int l_settbg(lua_State* L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);

	if(lua_isboolean(L, 2))
	{
		if(lua_toboolean(L, 2))
		{
			d->setTransparentBackgound(1.0);
		}
		else
		{
			d->setTransparentBackgound(0.0);
		}
	}
	else
	{
		d->setTransparentBackgound(lua_tonumber(L, 2));
	}
	return 0;

}

static int l_zoom(lua_State* L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	int z = lua_tointeger(L, 2);
	if(z < 0)
	{
		d->widget()->zoomOut(-z);
	}
	if(z > 0)
	{
		d->widget()->zoomIn( z);
	}
	return 0;
}

static int l_setfontfamily(lua_State* L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	QFont f(lua_tostring(L, 2));

	d->widget()->setFont(f);

	return 0;
}

static int l_sethighlighter(lua_State* L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	if(lua_toboolean(L, 2)) //true
	{
		if(d->highlighter)
			return 0;
		d->highlighter = new QLuaHilighter(d->widget()->document());
		//cerr << d->widget()->fontFamily().toStdString() << endl;
		d->widget()->setFontFamily("Courier");
	}
	else
	{
		if(d->highlighter)
			delete d->highlighter;
		d->widget()->setFontFamily("");

	}
	return 0;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* QTextEditItemLua::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setText",           l_settext},
		{"setHTML",           l_sethtml},
		{"addText",           l_addtext},
		{"addHTML",           l_addhtml},
		{"text",              l_gettext},
		{"setFrame",          l_setframe},
		{"setSize",           l_setsize},
		{"setWidth",          l_setwidth},
		{"setHeight",         l_setheight},
		{"setReadOnly",       l_setreadonly},
		{"zoom",              l_zoom},
		{"setHighlighter",      l_sethighlighter},
		{"setFontFamily",      l_setfontfamily},
		{"setTransparentBackground", l_settbg},
		//{"item",              l_item},
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

