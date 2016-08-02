#ifndef OPENGLSCENE_H
#define OPENGLSCENE_H

#include <QGraphicsScene>
extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}


class OpenGLScene : public QGraphicsScene
{
    Q_OBJECT
public:
    explicit OpenGLScene(QObject *parent = 0);
	~OpenGLScene();

	lua_State* L;
	int draw_func;
	void registerFunctions(lua_State* L);
protected:
	void drawBackground(QPainter *painter, const QRectF& );

signals:

public slots:

};

#endif // OPENGLSCENE_H
