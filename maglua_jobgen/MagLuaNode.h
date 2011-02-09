#ifndef MAGLUANODE_H
#define MAGLUANODE_H

#include <QPixmap>

class MagLuaNode
{
public:
	enum NodeType
	{
		Number,
		Operator,
		Data,
		Function,
		String
	};

	MagLuaNode(NodeType t=Data, QString n="VariableName", QString o="SpinSystem");

	NodeType type;
	QPixmap pixmap;
	QString name;
	QString objectname;
};

#endif // MAGLUANODE_H
