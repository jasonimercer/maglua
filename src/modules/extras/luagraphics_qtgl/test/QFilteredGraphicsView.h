#ifndef QFILTEREDGRAPHICSVIEW_H
#define QFILTEREDGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QMouseEvent>
extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}


class QFilteredGraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
	explicit QFilteredGraphicsView(QWidget *parent = 0);
	~QFilteredGraphicsView();

	int dragStartX, dragStartY;
	int dragEndX, dragEndY;
	int wheelDelta;
	float mx, my; /* mouse x, mouse y */

	lua_State* L;

	void setDrawFunction(int func);
	void setClickFunction(int func);
	void setKeyPressFunction(int func);

	int drawFunction;
	int clickFunction;
	int keyPressFunction;

protected:
	bool eventFilter(QObject *ob, QEvent *e);

	void resizeEvent(QResizeEvent* event);
	void wheelEvent(QWheelEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void mousePressEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void keyPressEvent(QKeyEvent *);

	//Ray MouseRay;



	float dragStartRR; //ratio about distance from center
	float dragStartRX;
	float dragStartRY;

	int windowx, windowy;
	bool mousePressInWidget;

signals:
	void keyPress(QKeyEvent* ke);

public slots:

};

#endif // QFILTEREDGRAPHICSVIEW_H
