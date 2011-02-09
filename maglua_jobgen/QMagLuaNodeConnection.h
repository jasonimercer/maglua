#ifndef QMAGLUANODECONNECTION_H
#define QMAGLUANODECONNECTION_H

class QMagLuaGraphicsNode;

class QMagLuaNodeConnection
{
public:
    QMagLuaNodeConnection();

	QMagLuaGraphicsNode* src;
	QMagLuaGraphicsNode* dest;

	int src_idx;
	int dest_idx;
};

#endif // QMAGLUANODECONNECTION_H
