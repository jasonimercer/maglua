#include "QMagLuaGraphicsNode.h"
#include <QLinearGradient>
#include <QtGui>
#include <QLineEdit>

QMagLuaGraphicsNode::QMagLuaGraphicsNode(MagLuaNode* _node, QGraphicsProxyWidget* _txtName)
	:	txtName(_txtName)
{
	setRect(0, 0, 200, 100);
	setFlags(QGraphicsItem::ItemIsFocusable|QGraphicsItem::ItemIsMovable|QGraphicsItem::ItemIsSelectable);
	setMagLuaNode(_node);

	if(txtName)
	{
		txtName->setParentItem(this);
		txtName->setPos(52,22);
		txtName->setMaximumWidth(128);
		QLineEdit* t = static_cast<QLineEdit*>(txtName->widget());
		if(t)
		{
			t->setText(node->name);
			t->setAlignment(Qt::AlignHCenter);
		}
	}

	icon = new QGraphicsSvgItem("gtk-info.svg", this);

	radius = 15;
	QRectF r = rect();
	r2 = r.adjusted(0, 0, -5, -5);

	topBanner = r2;
	topBanner.setHeight(radius);
	topBanner.adjusted(1, 0, -1, 0);

	bottomBanner = r2;
	bottomBanner.adjust(1, r2.height()-radius, -1, 0);

	centerBanner = r2;
	centerBanner.adjust(1, radius, -1, -radius);

	iconRect = centerBanner;
	iconRect.setWidth(iconRect.height());

	namesRect = centerBanner;
	namesRect.setLeft(iconRect.right());

	if(txtName)
	{
		txtName->setPos(namesRect.left() + 5, namesRect.top()+5);
		txtName->setMaximumWidth(namesRect.width() - 10);
	}

	if(icon)
	{
		QRectF ib = icon->boundingRect();
		float scale = (iconRect.width() - 10.0) / ib.width();
		icon->setScale(scale);
		icon->setPos(iconRect.left()+5, iconRect.top()+5);
	}
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

	QPen pen = p->pen();

	// Draw the node shadow
	if(opt->levelOfDetail >= 0.75)
	{
		QColor color1 = opt->palette.shadow().color();
		color1.setAlphaF(selected() ? 0.4 : 0.3);

		int shadowSize = selected() ? 5 : 3;
		QPainterPath path;
		path.addRoundRect(r2.adjusted(shadowSize,shadowSize,shadowSize,shadowSize), radius, radius*(rect().width() / rect().height()));
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


//	if(opt->levelOfDetail >= 0.75)
//	{
		QColor white = QColor(255,255,255);

		QPainterPath path;
		path.addRoundRect(r2.adjusted(1,1,-1,-1), radius, radius*(rect().width() / rect().height()));
		p->fillPath(path, nodeColor);

		QPainterPath p2;
		p2.addRect(centerBanner);
		p->fillPath(p2, white);

		p->setPen(QColor(0,0,0));
		p->drawPath(path);
		p->drawPath(p2);
//	}
//	else
//	{
//		QColor fillColor = opt->palette.light().color();
//		fillColor.setAlphaF(alpha);

//		QPainterPath path;
//		path.addRoundRect(r2.adjusted(1,1,-1,-1), radius, radius);
//		p->fillPath(path, fillColor);
//		p->drawPath(path);
//	}

	// Draw the node icon
//	QRectF textRect;
//	if(opt->levelOfDetail >= 0.75)
//	{
//		QPixmap img = QPixmap(":/gfx/info.png");//d->node->nodeDesc()->nodeIcon().pixmap(30, 30);
//		QRectF iconRect( r2.left()+10, r2.top()+10, img.width(), img.height() );
//		iconRect.moveTop( r2.center().y() - img.height()/2 );
//		p->drawPixmap( iconRect, img, QRectF(0,0,img.width(),img.height()) );

//		textRect = QRectF ( iconRect.right()+2, r2.top()+10, r2.width(), r2.height()-20 );
//		textRect.setRight( r2.right() );
//	}
//	else
//		textRect = r2;

	// Draw the node text
	p->setPen(pen);

//	if(opt->levelOfDetail >= 0.75)
//	{
		// First draw the node name
		//textRect.setBottom( r2.center().y() );
		//p->drawText(textRect, Qt::AlignCenter, node->name);//d->node->nodeName());

		// Now draw the node class name in a smaller font
//		QFont font = p->font();
//		QRectF textRect = r2.intersected(QRectF(0, r2.center().y(), r2.width(), r2.height()));
		//textRect.moveTop( r2.center().y() );
		p->drawText(namesRect.adjusted(0, namesRect.height()/2, 0, 0), Qt::AlignCenter, node->objectname);//d->node->nodeDesc()->nodeClassName());
//		p->setFont(font);
//	}
//	else
//		p->drawText(textRect, Qt::AlignCenter, node->objectname);//d->node->nodeDesc()->nodeClassName());

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

	//txtName->paint(p, opt, widget);
	//void QMagLuaGraphicsNode::paint(QPainter *p, const QStyleOptionGraphicsItem* opt, QWidget *widget)

}

