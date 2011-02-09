#include "QMagLuaGraphicsNode.h"
#include <QLinearGradient>
#include <QtGui>

QMagLuaGraphicsNode::QMagLuaGraphicsNode()
{
	setRect(0, 0, 200, 100);
		setFlags(QGraphicsItem::ItemIsFocusable|QGraphicsItem::ItemIsMovable|QGraphicsItem::ItemIsSelectable);
}

QMagLuaGraphicsNode::~QMagLuaGraphicsNode()
{
}

bool QMagLuaGraphicsNode::selected()
{
	return true;
}


void QMagLuaGraphicsNode::paint(QPainter *p, const QStyleOptionGraphicsItem* opt, QWidget *widget)
{
	Q_UNUSED(widget);

//	if(!d->node)
//		return;

	QRectF r = rect();
	QPen pen = p->pen();

#ifdef USE_SYSTEM_STYLE
	QStyleOptionButton hopt;
	hopt.rect = r.toRect();
	hopt.palette = opt->palette;
	hopt.state = QStyle::State_Active|QStyle::State_Enabled|QStyle::State_Horizontal|QStyle::State_Enabled|QStyle::State_Raised;
	if(selected())
		hopt.features = QStyleOptionButton::DefaultButton;
	if(widget)
		widget->style()->drawControl(QStyle::CE_PushButtonBevel, &hopt, p, 0);
	else
		QApplication::style()->drawControl(QStyle::CE_PushButtonBevel, &hopt, p, 0);

	p->setPen(pen);
	p->drawText(r2, Qt::AlignCenter, d->node->nodeName());
#else
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

	// Draw the node rectangle.
	double alpha = 0.60;
	if(selected())
		alpha = 0.95;

	if(selected())
		p->setPen( QPen(opt->palette.highlight().color(), 2) );
	else
	{
		QColor penColor = opt->palette.highlight().color();
		penColor.setAlphaF(0.85);
		p->setPen( QPen(penColor, 2) );
	}

	if(opt->levelOfDetail >= 0.75)
	{
		QColor topColor = opt->palette.highlight().color();
		QColor midColor = opt->palette.light().color();
		QColor bottomColor = topColor;

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
		QPixmap nodePm = QPixmap(":/gfx/info.png");//d->node->nodeDesc()->nodeIcon().pixmap(30, 30);
		QRectF iconRect( r2.left()+10, r2.top()+7, nodePm.width(), nodePm.height() );
		iconRect.moveTop( r2.center().y() - nodePm.height()/2 );
		p->drawPixmap( iconRect, nodePm, QRectF(0,0,nodePm.width(),nodePm.height()) );
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
		p->drawText(textRect, Qt::AlignCenter, "123456");//d->node->nodeName());

		// Now draw the node class name in a smaller font
		QFont font = p->font();
		QFont newFont = font;
		newFont.setPointSize( font.pointSize()-1 );
		p->setFont( newFont );
		textRect.moveTop( r2.center().y()+1 );
		p->drawText(textRect, Qt::AlignCenter, "classname");//d->node->nodeDesc()->nodeClassName());
		p->setFont(font);
	}
	else
		p->drawText(textRect, Qt::AlignCenter, "classname");//d->node->nodeDesc()->nodeClassName());

#endif

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

