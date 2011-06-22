#include "LuaThread.h"
#include <QTextEdit>
#include <stdio.h>

LuaThread::LuaThread(QObject *parent) :
    QThread(parent)
{
	L = 0;
	requestStop = false;
}

LuaThread::~LuaThread()
{
	mutex.lock();
//	abort = true;
	condition.wakeOne();
	mutex.unlock();

}

bool LuaThread::stopRequested() const
{
	return requestStop;
}


void LuaThread::PrintOutput(const QString& text)
{
	emit(printOutput(text));
}

void LuaThread::PrintError(const QString& text)
{
	emit(printError(text));
}



static int print(lua_State* L)
{
	LuaThread* thread = (LuaThread*)lua_touserdata(L, lua_upvalueindex(1));
	if(!thread)
		return luaL_error(L, "LuaThread upvalue is not set");

	QString line = "";

	int n = lua_gettop(L);  /* number of arguments */
	int i;
	lua_getglobal(L, "tostring");
	for(i=1; i<=n; i++)
	{
		const char *s;
		lua_pushvalue(L, -1);
		lua_pushvalue(L, i);
		lua_call(L, 1, 1);
		s = lua_tostring(L, -1);  /* get result */
		if (s == NULL)
			return luaL_error(L, LUA_QL("tostring") " must return a string to "
							  LUA_QL("print"));
		if (i>1)
			line.append("\t");
		line.append(s);
		lua_pop(L, 1);  /* pop result */
	}
	line.append("\n");

	thread->PrintOutput(line);

	return 0;
}

static void l_handler(lua_State *L, lua_Debug* /* ar */)
{
	lua_rawgeti(L, LUA_REGISTRYINDEX, 100);
	LuaThread* thread = (LuaThread*) lua_touserdata(L, -1);
	lua_pop(L, 1);
	if(thread)
	{
		if(thread->stopRequested())
		{
			luaL_error(L, "Stopped by user request");
		}
	}
}

void LuaThread::run()
{
	if(lua_pcall(L, 0, 0, -2))
	{
		printf("%s\n", lua_tostring(L, -1));
		emit(printError(lua_tostring(L, -1)));
	}

	mutex.lock();
	lua_close(L);
	L = 0;
	mutex.unlock();
}

void LuaThread::stop()
{
	requestStop = true;
}

void LuaThread::execute(lua_State* LL)
{
	L = LL;

	lua_pushlightuserdata(L, this);
	lua_pushcclosure(L, print, 1);
	lua_setglobal(L, "print");

	lua_pushlightuserdata(L, this);
	lua_rawseti(L, LUA_REGISTRYINDEX, 100);

	lua_sethook(L, l_handler, LUA_MASKCOUNT, 10);

//	QMutexLocker locker(&mutex);

	requestStop = false;

	if(!isRunning())
	{
		start(LowPriority);
	}
	else
	{
//		restart = true;
//		condition.wakeOne();
	}

//	lua_close(L);
}
