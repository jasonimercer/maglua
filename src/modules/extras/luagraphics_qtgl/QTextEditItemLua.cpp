#include "QTextEditItemLua.h"
#include <QErrorMessage>
#include <iostream>
#include <QApplication>

#include "QGraphicsSceneLua.h"
using namespace std;

QTextEditItemLua::QTextEditItemLua()
	: QItemLua(hash32(lineage(0)))
{
	textedit = 0;
	highlighter = 0;
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
	QItemLua::luaInit(L);
	if(!scene) return 0;

	textedit = new QTextEdit(QApplication::activeWindow());
	//textedit = new QTextEdit();
	textedit->show();
	textedit->setTabStopWidth(40);

	proxy = scene->addWidget(textedit, 0);
	proxy->setPos(0,0);
	proxy->show();

	setTransparent(1);

	return 0;
}

void QTextEditItemLua::push(lua_State* L)
{
	luaT_push<QTextEditItemLua>(L, this);
}

void QTextEditItemLua::setTransparent(float t)
{
	if(!textedit) return;

	textedit->setAttribute(Qt::WA_NoSystemBackground, true);
	QPalette pal = textedit->palette();
	pal.setBrush(QPalette::Base, QColor(255,255,255,255*(1-t)));
	textedit->setPalette(pal);
}












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

static int l_setsb(lua_State* L)
{
	LUA_PREAMBLE(QTextEditItemLua, d, 1);
	if(!d->widget()) return 0;

	int i = lua_tointeger(L, 2);

	if(i == 0)
	{
		d->widget()->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		d->widget()->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	}

	if(i == 1)
	{
		d->widget()->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		d->widget()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	}

	if(i == 2)
	{
		d->widget()->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
		d->widget()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	}
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

const luaL_Reg* QTextEditItemLua::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)return m;

	merge_luaL_Reg(m, QItemLua::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setText",           l_settext},
		{"setHTML",           l_sethtml},
		{"addText",           l_addtext},
		{"addHTML",           l_addhtml},
		{"text",              l_gettext},
		{"setFrame",          l_setframe},
		{"setReadOnly",       l_setreadonly},
		{"zoom",              l_zoom},
		{"setHighlighter",    l_sethighlighter},
		{"setFontFamily",     l_setfontfamily},
		{"setScrollBarPolicy",l_setsb},
		{NULL,NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

