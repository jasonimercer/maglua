#include "QMagLuaGraphicsView.h"

QMagLuaGraphicsView::QMagLuaGraphicsView(QWidget *parent) :
    QGraphicsView(parent)
{
//    d->scene->setItemIndexMethod(QGraphicsScene::NoIndex);
//    d->scene->installEventFilter(this);
//	d->scene->setSceneRect(-5000, -5000, 10000, 10000);
	setRenderHints(QPainter::Antialiasing|QPainter::TextAntialiasing|QPainter::SmoothPixmapTransform);
	_zoomLevel = 0;
}

void QMagLuaGraphicsView::wheelEvent(QWheelEvent* e)
{
	int delta = e->delta() / 120;
	setZoomLevel(delta + zoomLevel());
	e->accept();
}

void QMagLuaGraphicsView::zoomIn()
{
	setZoomLevel(zoomLevel() + 1);
}

void QMagLuaGraphicsView::zoomOut()
{
	setZoomLevel(zoomLevel() - 1);
}

void QMagLuaGraphicsView::zoomReset()
{
	setZoomLevel(0);
}

int QMagLuaGraphicsView::zoomLevel() const
{
	return _zoomLevel;
}

int QMagLuaGraphicsView::zoomMin() const
{
	return -10;
}

int QMagLuaGraphicsView::zoomMax() const
{
	return 10;
}

static void clamp(int& v, int min, int max)
{
	if(v > max) v = max;
	if(v < min) v = min;
}

void QMagLuaGraphicsView::setZoomLevel(int zl)
{
	clamp(zl, zoomMin(), zoomMax());

	if(zoomLevel() == zl)
		return;

	_zoomLevel = zl;
	//d->zoomValue = d->calculateScaleFor(zl);

	// Set the zoom to the graphics view.
	QApplication::setOverrideCursor(Qt::WaitCursor);
	QMatrix matrix;
	matrix.scale(pow(1.1, zl), pow(1.1, zl));
	setMatrix(matrix);
	QApplication::restoreOverrideCursor();

//    emit zoomLevelChanged(d->zoomLevel);
//    emit zoomScaleChanged(d->zoomValue);
//    emit zoomEvent(d->zoomValue);
}
