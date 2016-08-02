#ifndef QMAGLUAGRAPHICSVIEW_H
#define QMAGLUAGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QtGui>

class QMagLuaGraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
	explicit QMagLuaGraphicsView(QWidget *parent = 0);

	int zoomLevel() const;
	void setZoomLevel(int zl);
	int zoomMin() const;
	int zoomMax() const;

	void zoomReset();
	void zoomIn();
	void zoomOut();

signals:

public slots:

protected:
	void wheelEvent(QWheelEvent* e);

	int _zoomLevel;

};

#endif // QMAGLUAGRAPHICSVIEW_H
