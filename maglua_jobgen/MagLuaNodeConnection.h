#ifndef MAGLUANODECONNECTION_H
#define MAGLUANODECONNECTION_H

#include <QString>

class MagLuaNode;

class MagLuaNodeConnection
{
public:
    MagLuaNodeConnection();

	MagLuaNode* src;
	MagLuaNode* dest;

	QString src_name;
	QString dest_name;

	int src_idx;
	int dest_idx;
};

#endif // MAGLUANODECONNECTION_H
