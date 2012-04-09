#include "QFilteredGraphicsView.h"
#include "MainWindow.h"
#include <QErrorMessage>
#include <QDebug>
#include <QGraphicsSvgItem>
#include <QGraphicsProxyWidget>
#include <QApplication>

#include <math.h>
#include <iostream>
using namespace std;
QFilteredGraphicsView::QFilteredGraphicsView(QWidget *parent) :
	QGraphicsView(parent)
{
	installEventFilter(this);
	L = 0;
	key_func = LUA_REFNIL;

	dragStartX  = 0;
	dragStartY  = 0;

	dragEndX  = 0;
	dragEndY  = 0;

	mousePressInWidget = false;
}


static int l_centerOn(lua_State* L)
{
	void* v = lua_touserdata(L, lua_upvalueindex(1));
	QFilteredGraphicsView* vv = (QFilteredGraphicsView*)v;
	vv->centerOn(
				lua_tonumber(L, 1),
				lua_tonumber(L, 2)
				);
	return 0;
}
static int l_setkeyfunc(lua_State* L)
{
	void* v = lua_touserdata(L, lua_upvalueindex(1));
	QFilteredGraphicsView* vv = (QFilteredGraphicsView*)v;

	if(lua_isfunction(L, -1))
	{
		luaL_unref(L, LUA_REGISTRYINDEX, vv->key_func);
		vv->key_func = luaL_ref(L, LUA_REGISTRYINDEX);
	}
	else
	{
		vv->key_func = LUA_REFNIL;
	}

	return 0;
}


void QFilteredGraphicsView::registerFunctions(lua_State* _L)
{
	L = _L;
	lua_pushlightuserdata(L, (void*)this);
	lua_pushcclosure(L, l_centerOn, 1);
	lua_setglobal(L, "centerOn");

	lua_pushlightuserdata(L, (void*)this);
	lua_pushcclosure(L, l_setkeyfunc, 1);
	lua_setglobal(L, "setKeyFunction");
}


QFilteredGraphicsView::~QFilteredGraphicsView()
{
	if(L)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, key_func);
	}
}

bool QFilteredGraphicsView::eventFilter(QObject *ob, QEvent *e)
{
	return QGraphicsView::eventFilter(ob, e);
}







void QFilteredGraphicsView::wheelEvent(QWheelEvent* event)
{
	QGraphicsView::wheelEvent(event);
}

void QFilteredGraphicsView::keyPressEvent(QKeyEvent* event )
{
	if(!scene())
		return QGraphicsView::keyPressEvent(event);

	QGraphicsItem* fi = scene()->focusItem();
	QGraphicsProxyWidget* pw = dynamic_cast<QGraphicsProxyWidget*>(fi);
//	if(!pw)
//		return QGraphicsView::keyPressEvent(event);
	//if(! scene()->focusItem() && key_func != LUA_REFNIL)
	if(!pw && key_func != LUA_REFNIL)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, key_func);
		QString t = event->text();

		switch(event->key())
		{
		case Qt::Key_Escape:	t = "escape"; break;
		case Qt::Key_Tab:		t = "tab"; break;
		case Qt::Key_Backspace:	t = "backspace"; break;
		case Qt::Key_Return:	t = "return"; break;
		case Qt::Key_Enter:		t = "enter"; break;
		case Qt::Key_Insert:	t = "insert"; break;
		case Qt::Key_Delete:	t = "delete"; break;
		case Qt::Key_Print:		t = "print"; break;
		case Qt::Key_Clear:		t = "clear"; break;
		case Qt::Key_Home:		t = "home"; break;
		case Qt::Key_End:		t = "end"; break;
		case Qt::Key_Left:		t = "left"; break;
		case Qt::Key_Up:		t = "up"; break;
		case Qt::Key_Right:		t = "right"; break;
		case Qt::Key_Down:		t = "down"; break;
		case Qt::Key_PageUp:	t = "pageup"; break;
		case Qt::Key_PageDown:	t = "pagedown"; break;

		case Qt::Key_Shift:		t = "shift"; break;
		case Qt::Key_Control:	t = "control"; break;
		case Qt::Key_Alt:		t = "alt"; break;
		}

		lua_pushstring(L, t.toStdString().c_str());
		if(lua_pcall(L, 1, 1, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
			QErrorMessage* msg = new QErrorMessage(QApplication::activeWindow());
			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
			lua_pop(L, lua_gettop(L));
		}

		bool b = lua_toboolean(L, -1);
		lua_gc(L, LUA_GCCOLLECT, 0);

//		if(!b)
//			QGraphicsView::keyPressEvent(event);
	}
	else
	{
		QGraphicsView::keyPressEvent(event);
	}
}

static void clampf(float& val, float minimum, float maximum)
{
	if(val < minimum) val = minimum;
	if(val > maximum) val = maximum;
}

void QFilteredGraphicsView::mouseMoveEvent(QMouseEvent* event)
{
	QGraphicsView::mouseMoveEvent(event);
}


void QFilteredGraphicsView::mousePressEvent(QMouseEvent* event)
{
	QGraphicsView::mousePressEvent(event);
}

void QFilteredGraphicsView::mouseReleaseEvent(QMouseEvent* event)
{
	mousePressInWidget = false;
	QGraphicsView::mouseReleaseEvent(event);
}

void QFilteredGraphicsView::resizeEvent(QResizeEvent* event)
{
	// setup viewport, projection etc.:
	//glViewport(0, 0, (GLint)w, (GLint)h);
	if(L)
	{
		lua_pushinteger(L, event->size().width());
		lua_setglobal(L, "window_x");
		lua_pushinteger(L, event->size().height());
		lua_setglobal(L, "window_y");
	}

	//QGraphicsView::resizeEvent(event);
}


