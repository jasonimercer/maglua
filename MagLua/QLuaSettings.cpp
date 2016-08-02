#include "QLuaSettings.h"
#include "MainWindow.h"
#include <QSettings>

static int l_stringList(lua_State* L)
{
	QSettings settings(ORGANIZATION, APPLICATION);

	if(lua_gettop(L) == 1)
	{
		QStringList list = settings.value(lua_tostring(L, 1), QStringList()).toStringList();

		lua_newtable(L);
		for(int i=0; i<list.size(); i++)
		{
			lua_pushinteger(L, i+1);
			lua_pushstring(L, list.at(i).toStdString().c_str());
			lua_settable(L, -3);
		}
		return 1;
	}

	QStringList list;

	if(lua_istable(L, 2))
	{
		lua_pushnil(L);
		while(lua_next(L, 2))
		{
			list << lua_tostring(L, -1);
			lua_pop(L, 1);
		}
	}
	else
	{
		if(lua_isstring(L, 2))
		{
			list << lua_tostring(L, 2);
		}
	}

	settings.setValue(lua_tostring(L, 1), list);

	return 0;
}

static int l_string(lua_State* L)
{
	QSettings settings(ORGANIZATION, APPLICATION);
	if(lua_gettop(L) == 1)
	{
		QString ss = settings.value(lua_tostring(L, 1), QString()).toString();

		lua_pushstring(L, ss.toStdString().c_str());

		return 1;
	}

	QString ss;

	if(lua_isstring(L, 2))
	{
		ss =  lua_tostring(L, -1);
	}

	settings.setValue(lua_tostring(L, 1), ss);

	return 0;
}

void lua_registerqluasettings(lua_State* L)
{
	lua_newtable(L);

	lua_pushcfunction(L, l_stringList);
	lua_setfield(L, -2, "stringList");

	lua_pushcfunction(L, l_string);
	lua_setfield(L, -2, "string");

	lua_setglobal(L, "QSettings");
}
