#ifndef QMAGLUAGRAPHICSNODE_H
#define QMAGLUAGRAPHICSNODE_H

#include <QObject>
#include <QGraphicsRectItem>


class QMagLuaGraphicsNode : public QObject, public QGraphicsRectItem
{
	Q_OBJECT
public:
	QMagLuaGraphicsNode();
	~QMagLuaGraphicsNode();

	bool selected();
protected:
	void paint(QPainter *p, const QStyleOptionGraphicsItem* opt, QWidget *widget);
};

#endif // QMAGLUAGRAPHICSNODE_H
