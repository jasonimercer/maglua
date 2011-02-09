#ifndef QMAGLUAGRAPHICSNODE_H
#define QMAGLUAGRAPHICSNODE_H

#include <QObject>
#include <QGraphicsRectItem>
#include <MagLuaNode.h>
#include <QtGui>

class QMagLuaGraphicsNode : public QObject, public QGraphicsRectItem
{
	Q_OBJECT
public:
	QMagLuaGraphicsNode(MagLuaNode* _node=0);
	~QMagLuaGraphicsNode();

	bool selected();
	void setMagLuaNode(MagLuaNode* node);
protected:
	void paint(QPainter *p, const QStyleOptionGraphicsItem* opt, QWidget *widget);
	MagLuaNode* node;

	QColor lookupNodeColorType(MagLuaNode::NodeType t);
};

#endif // QMAGLUAGRAPHICSNODE_H
