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
	//installEventFilter(this);
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
#if 0
	if(mousePressInWidget) return;

	Camera* cam = Singleton.mainWindow->cam;
	if(!cam) return;

	int x = event->x() - dragStartX;
	int y = event->y() - dragStartY;

	float m = width();
	if(height() < m)
		m = height();

	float vx = event->x() - width() / 2;
	float vy = event->y() - height() / 2;

	vx /= (m*0.5);
	vy /= (m*0.5);

	clampf(vx, -1, 1);
	clampf(vy, -1, 1);

	double fx = ((float)x) / m;
	double fy = ((float)y) / m;

	if((event->modifiers() & Qt::ShiftModifier) == Qt::ShiftModifier)
	{
		if((event->buttons() & Qt::LeftButton) == Qt::LeftButton)
		{
			fx *=-cam->dist();
			fy *= cam->dist();

			cam->translateUVW(Vector(fx, fy, 0));
		}
	}
	else
	{
		fx *= 2;
		fy *= 2;

		if((event->buttons() & Qt::LeftButton) == Qt::LeftButton)
		{
			cam->rotateAbout(-fy * vx, cam->forward);
			cam->rotateAbout(-fx * (1.0 - fabs(vy)), cam->up);

			cam->rotateAbout( fx * vy, cam->forward);
			cam->rotateAbout(-fy * (1.0 - fabs(vx)), cam->right);

		}
	}

	dragStartX = event->x();
	dragStartY = event->y();

	double dx = 2.0 * ((double)event->x()) / ((double)width())  - 1.0;
	double dy = 2.0 * ((double)event->y()) / ((double)height()) - 1.0;

	if(cam->perspective)
	{
		MouseRay.origin = cam->pos;
		MouseRay.direction = ((cam->forward) +
							  (tan( cam->FOV * 0.505 * dx * 3.1415926/180.0 * cam->ratio) * cam->right) -
							  (tan( cam->FOV * 0.505 * dy * 3.1415926/180.0 ) * cam->up));
	}
	else
	{
		const float d = cam->dist();
		const float xx = d*0.5*cam->ratio;
		const float yy =-d*0.5;

		MouseRay.origin = cam->pos + cam->right * xx * dx + cam->up * yy * dy;
		MouseRay.direction = cam->forward;
	}
#endif
}


void QFilteredGraphicsView::mousePressEvent(QMouseEvent* event)
{
	//	QGraphicsItem* item = itemAt(event->x(), event->y());
	//	if(item)
	//	{
	//		QGraphicsView::mousePressEvent(event);
	//		//item->setFocus(Qt::MouseFocusReason);

	//		if(event->isAccepted())
	//			return;
	//	}

#if 0
	mousePressInWidget = false;
	if(item)
	{
		QGraphicsProxyWidget* qgpw = dynamic_cast<QGraphicsProxyWidget*>(item);
		QGraphicsSvgItem* qgsi = dynamic_cast<QGraphicsSvgItem*>(item);

		// allow writable text edits to catch the mouse move
		if(qgpw)
		{
			QTextEdit* te = dynamic_cast<QTextEdit*>(qgpw->widget());
			if(te && !te->isReadOnly())
			{
				qgpw = 0;
			}
		}


		//qDebug() << itemAt(event->x(), event->y()) << endl;

		if(!qgpw && !qgsi)
		{
			mousePressInWidget = true;
			QGraphicsView::mousePressEvent(event);

			//if(event->isAccepted())
			return;
		}
	}


	//	cout << event->x() << ", " << event->y() << endl;

	// allow writable text edits to scroll on mouse wheel
	//	if(qgpw)
	//	{
	//		QTextEdit* te = dynamic_cast<QTextEdit*>(qgpw->widget());
	//		if(te && !te->isReadOnly())
	//		{
	//			qgpw = 0;
	//		}
	//	}


	dragStartX = event->x();
	dragStartY = event->y();


	qreal xx = event->x();
	qreal yy = event->y();

	qreal cx = width() / 2.0;
	qreal cy = height() / 2.0;

	qreal rx = cx-xx;
	qreal ry = cy-yy;

	qreal rr = sqrt(rx*rx+ry*ry);

	if(cx > cy)
	{
		rr /= (cx * 0.75);
		rx /= (cx * 0.75);
		ry /= (cx * 0.75);
	}
	else
	{
		rr /= (cy * 0.75);
		rx /= (cy * 0.75);
		ry /= (cy * 0.75);
	}

	if(rr > 1.0)
		rr = 1.0;

	dragStartRR = rr;
	dragStartRX = rx;
	dragStartRY = ry;


	if(hasClickFunction && L)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, clickFunction);

		lua_pushboolean(L, event->buttons() & Qt::LeftButton);
		lua_pushboolean(L, event->buttons() & Qt::MidButton);
		lua_pushboolean(L, event->buttons() & Qt::RightButton);
		lua_pushray(L, new Ray(MouseRay));

		if(lua_pcall(L, 4, 0, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
			QErrorMessage* msg = new QErrorMessage(this);
			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
			lua_pop(L, lua_gettop(L));
		}
		lua_gc(L, LUA_GCCOLLECT, 0);
	}
#endif
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


