#include "OpenGLScene.h"
#include <QPainter>
#include <QPaintEngine>
#include <QGLWidget>
#include "MainWindow.h"
#include <QErrorMessage>

OpenGLScene::OpenGLScene(QObject *parent) :
		QGraphicsScene(parent)
{
	L = 0;
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
	//lua_State* L = Singleton.mainWindow->L;

	glClearColor(1,1,1, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(L)// && Singleton.mainWindow->ui->view->hasDrawFunction)
	{
		glPushAttrib(GL_COLOR_BUFFER_BIT);
		glPushAttrib(GL_CURRENT_BIT);
		glPushAttrib(GL_ENABLE_BIT);
		glPushAttrib(GL_LIGHTING_BIT);
		glPushAttrib(GL_POLYGON_BIT);
		glPushAttrib(GL_TRANSFORM_BIT);
		glPushAttrib(GL_VIEWPORT_BIT);

		/*
		lua_rawgeti(L, LUA_REGISTRYINDEX, Singleton.mainWindow->ui->view->drawFunction);
		if(lua_pcall(L, 0, 0, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
			QErrorMessage* msg = new QErrorMessage(Singleton.mainWindow);
			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
			lua_pop(L, lua_gettop(L));
		}
		lua_gc(L, LUA_GCCOLLECT, 0);
		*/
		for(int i=0; i<7; i++)
		{
			glPopAttrib();
		}
	}
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
}
