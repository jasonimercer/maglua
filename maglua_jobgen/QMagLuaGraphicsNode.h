#ifndef QMAGLUAGRAPHICSNODE_H
#define QMAGLUAGRAPHICSNODE_H

#include <QObject>
#include <QGraphicsRectItem>
#include <MagLuaNode.h>
#include <QtGui>
#include <QGraphicsProxyWidget>
#include <QGraphicsSvgItem>

class QMagLuaGraphicsNode : public QObject, public QGraphicsRectItem
{
	Q_OBJECT
public:
	QMagLuaGraphicsNode(MagLuaNode* _node=0, QGraphicsProxyWidget* _txtName=0);
	~QMagLuaGraphicsNode();

	bool selected();
	void setMagLuaNode(MagLuaNode* node);
protected:
	void paint(QPainter *p, const QStyleOptionGraphicsItem* opt, QWidget *widget);
	MagLuaNode* node;

	QColor lookupNodeColorType(MagLuaNode::NodeType t);
	QGraphicsProxyWidget* txtName;
	QGraphicsSvgItem* icon;

	int radius; //rounded corners
	QRectF  r2;
	QRectF topBanner;
	QRectF bottomBanner;
	QRectF centerBanner;
	QRectF iconRect;
	QRectF namesRect;
};

#endif // QMAGLUAGRAPHICSNODE_H
