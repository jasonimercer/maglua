#include "QMagLuaGraphicsNode.h"
#include <QLinearGradient>
#include <QtGui>

QMagLuaGraphicsNode::QMagLuaGraphicsNode(MagLuaNode* _node)
{
	setRect(0, 0, 200, 100);
	setFlags(QGraphicsItem::ItemIsFocusable|QGraphicsItem::ItemIsMovable|QGraphicsItem::ItemIsSelectable);
	setMagLuaNode(_node);


}

void QMagLuaGraphicsNode::setMagLuaNode(MagLuaNode* _node)
{
	node = _node;
}

QMagLuaGraphicsNode::~QMagLuaGraphicsNode()
{
}

bool QMagLuaGraphicsNode::selected()
{
	return true;
}

QColor QMagLuaGraphicsNode::lookupNodeColorType(MagLuaNode::NodeType t)
{
	switch(t)
	{
	case MagLuaNode::Data:		return QColor(255,0,0);
	case MagLuaNode::Function:	return QColor(255,255,0);
	case MagLuaNode::Number:	return QColor(0,255,0);
	case MagLuaNode::Operator:	return QColor(0,0,255);
	case MagLuaNode::String:	return QColor(128,128,128);
	}
	return QColor(0,0,0);
}


void QMagLuaGraphicsNode::paint(QPainter *p, const QStyleOptionGraphicsItem* opt, QWidget *widget)
{
	Q_UNUSED(widget);

	if(!node)
		return;

	QRectF r = rect();
	QPen pen = p->pen();
	QRectF r2 = r.adjusted(0, 0, -5, -5);

	// Draw the node shadow
	if(opt->levelOfDetail >= 0.75)
	{
		QColor color1 = opt->palette.shadow().color();
		color1.setAlphaF(selected() ? 0.4 : 0.3);

		int shadowSize = selected() ? 5 : 3;
		QPainterPath path;
		path.addRoundRect(r2.adjusted(shadowSize,shadowSize,shadowSize,shadowSize), 10, 10);
		p->fillPath(path, color1);
	}

	QColor nodeColor = lookupNodeColorType(node->type);

	// Draw the node rectangle.
	double alpha;
	if(selected())
	{
		alpha = 1.0;
		p->setPen( QPen(nodeColor, 2) );
	}
	else
	{
		alpha = 0.6;
		QColor c = nodeColor;
		c.setAlphaF(0.8);
		p->setPen( QPen(c, 2) );
	}




	if(opt->levelOfDetail >= 0.75)
	{
		QColor topColor = nodeColor;//opt->palette.highlight().color();
		QColor midColor = opt->palette.light().color();
		QColor bottomColor = topColor;
		QColor white = QColor(255,255,255);

		topColor.setAlphaF(alpha);
		midColor.setAlphaF(alpha);
		bottomColor.setAlphaF(alpha);

		QLinearGradient grad(r2.topLeft(), r2.bottomLeft());
		grad.setColorAt(0, topColor);
		grad.setColorAt(0.2, midColor);
		grad.setColorAt(0.8, midColor);
		grad.setColorAt(1, bottomColor);

		QPainterPath path;
		path.addRoundRect(r2.adjusted(1,1,-1,-1), 10, 10);
		p->fillPath(path, white);
		p->fillPath(path, grad);
		p->drawPath(path);
	}
	else
	{
		QColor fillColor = opt->palette.light().color();
		fillColor.setAlphaF(alpha);

		QPainterPath path;
		path.addRoundRect(r2.adjusted(1,1,-1,-1), 10, 10);
		p->fillPath(path, fillColor);
		p->drawPath(path);
	}

	// Draw the node icon
	QRectF textRect;
	if(opt->levelOfDetail >= 0.75)
	{
		QPixmap img = QPixmap(":/gfx/info.png");//d->node->nodeDesc()->nodeIcon().pixmap(30, 30);
		QRectF iconRect( r2.left()+10, r2.top()+7, img.width(), img.height() );
		iconRect.moveTop( r2.center().y() - img.height()/2 );
		p->drawPixmap( iconRect, img, QRectF(0,0,img.width(),img.height()) );

		textRect = QRectF ( iconRect.right()+2, r2.top()+10, r2.width(), r2.height()-20 );
		textRect.setRight( r2.right() );
	}
	else
		textRect = r2;

	// Draw the node text
	p->setPen(pen);

	if(opt->levelOfDetail >= 0.75)
	{
		// First draw the node name
		textRect.setBottom( r2.center().y() );
		p->drawText(textRect, Qt::AlignCenter, node->name);//d->node->nodeName());

		// Now draw the node class name in a smaller font
		QFont font = p->font();
		QFont newFont = font;
		newFont.setPointSize( font.pointSize()-1 );
		p->setFont( newFont );
		textRect.moveTop( r2.center().y()+1 );
		p->drawText(textRect, Qt::AlignCenter, node->objectname);//d->node->nodeDesc()->nodeClassName());
		p->setFont(font);
	}
	else
		p->drawText(textRect, Qt::AlignCenter, node->objectname);//d->node->nodeDesc()->nodeClassName());

	//	if(opt->levelOfDetail >= 0.75)
	//		d->node->paintNode(p, r, *opt);

	QBrush brush = opt->palette.mid();
	QColor color = brush.color();
	color.setAlphaF(0.75f);
	brush.setColor(color);
	color = opt->palette.shadow().color();
	color.setAlphaF(0.75f);

	//	QMap<IVisSystemNodeConnectionPath*, QRectF>::iterator it = d->pathRectMap.begin();
	//	QMap<IVisSystemNodeConnectionPath*, QRectF>::iterator end = d->pathRectMap.end();
	//	while(it != end)
	//	{
	//		p->setPen(color);
	//		p->setBrush(brush);
	//		p->drawRect(it.value());
	//		p->setPen(pen);
	//		d->node->paintConnectionPath(it.key(), p, it.value(), *opt);
	//		++it;
	//	}
}

