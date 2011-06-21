#ifndef LUATHREAD_H
#define LUATHREAD_H

#include <QMutex>
#include <QThread>
#include <QWaitCondition>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}


class LuaThread : public QThread
{
    Q_OBJECT
public:
    explicit LuaThread(QObject *parent = 0);
	~LuaThread();

	void PrintOutput(const QString& text);
	void PrintError(const QString& text);

	void execute(lua_State* L);
	void stop();
signals:
	void printOutput(const QString& text);
	void printError(const QString& text);

protected:
	void run();

private:
	lua_State* L;
	QMutex mutex;
	QWaitCondition condition;
};

#endif // LUATHREAD_H
