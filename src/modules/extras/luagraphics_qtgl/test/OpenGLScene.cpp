#include "OpenGLScene.h"
#include <QPainter>
#include <QPaintEngine>
#include <QGLWidget>
#include "MainWindow.h"
#include <QErrorMessage>
#include <iostream>
using namespace std;

OpenGLScene::OpenGLScene(QObject *parent) :
		QGraphicsScene(parent)
{
	L = 0;
	draw_func = LUA_REFNIL;

//	setSceneRect(0,0,1000,1000);
}

OpenGLScene::~OpenGLScene()
{
	if(L)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, draw_func);
	}
}


static int l_setSceneRect(lua_State* L)
{
	void* v = lua_touserdata(L, lua_upvalueindex(1));
	OpenGLScene* vv = (OpenGLScene*)v;
	vv->setSceneRect(
				lua_tonumber(L, 1),
				lua_tonumber(L, 2),
				lua_tonumber(L, 3),
				lua_tonumber(L, 4)
				);
	return 0;
}

static int l_setdrawfunc(lua_State* L)
{
	void* v = lua_touserdata(L, lua_upvalueindex(1));
	OpenGLScene* vv = (OpenGLScene*)v;

	if(lua_isfunction(L, -1))
	{
		luaL_unref(L, LUA_REGISTRYINDEX, vv->draw_func);
		vv->draw_func = luaL_ref(L, LUA_REGISTRYINDEX);
	}
	else
	{
		vv->draw_func = LUA_REFNIL;
	}
	return 0;
}

void OpenGLScene::registerFunctions(lua_State* _L)
{
	L = _L;
	lua_pushlightuserdata(L, (void*)this);
	lua_pushcclosure(L, l_setdrawfunc, 1);
	lua_setglobal(L, "setSceneDrawFunction");

	lua_pushlightuserdata(L, (void*)this);
	lua_pushcclosure(L, l_setSceneRect, 1);
	lua_setglobal(L, "setSceneRect");
}


void OpenGLScene::drawBackground(QPainter *painter, const QRectF& )
{
	if((painter->paintEngine()->type() != QPaintEngine::OpenGL ) &&
	   (painter->paintEngine()->type() != QPaintEngine::OpenGL2))
	{
		qWarning("OpenGLScene: drawBackground needs a "
				 "QGLWidget to be set as viewport on the "
				 "graphics view");
		return;
	}

	glPushMatrix();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//lua_State* L = Singleton.mainWindow->L;

	glClearColor(1,1,1, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(L && draw_func != LUA_REFNIL)// && Singleton.mainWindow->ui->view->hasDrawFunction)
	{
		glPushAttrib(GL_COLOR_BUFFER_BIT);
		glPushAttrib(GL_CURRENT_BIT);
		glPushAttrib(GL_ENABLE_BIT);
		glPushAttrib(GL_LIGHTING_BIT);
		glPushAttrib(GL_POLYGON_BIT);
		glPushAttrib(GL_TRANSFORM_BIT);
		glPushAttrib(GL_VIEWPORT_BIT);

		lua_rawgeti(L, LUA_REGISTRYINDEX, draw_func);
		if(lua_pcall(L, 0, 0, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
//			QErrorMessage* msg = new QErrorMessage(Singleton.mainWindow);
//			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
//			lua_pop(L, lua_gettop(L));
		}
		lua_gc(L, LUA_GCCOLLECT, 0);

		for(int i=0; i<7; i++)
		{
			glPopAttrib();
		}
	}
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
}
