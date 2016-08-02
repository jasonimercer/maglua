#ifndef LUATHREAD_H
#define LUATHREAD_H

#include <QMutex>
#include <QThread>
#include <QWaitCondition>
#include <QTextEdit>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}


class LuaThread : public QThread
{
    Q_OBJECT
public:
	explicit LuaThread(lua_State* parentL, QObject *parent = 0);
	~LuaThread();

	void setOutFunction(int ref);
	void setErrFunction(int ref);

	void callErr(const QString& msg);
	void callOut(const QString& msg);
	void doFile(const QString file, const QString args);
	void doCode(const QString code, const QString args);

	void execute();
	void stop();
	bool stopRequested() const;
	bool running() const {return L!=0;}
	void setCurrentLine(int line, const QString& src);

	int loadModules(lua_State* L);

	int refcount; //in parent

	int outputFuncRef;
	int errorFuncRef;

signals:
	void currentLineChange(int line, const QString& src);
	void threadedCallWithMsg(lua_State* L, int ref, const QString& msg);

protected:
	void run();

private:
	int preSetup(const QString args, const QString title);

	lua_State* parentL;

	QStringList loadList;

	lua_State* L;
	QMutex mutex;
	QWaitCondition condition;
	bool requestStop;

	int currentLine;
	QString currentSource;
};

#endif // LUATHREAD_H


int lua_isluathread(lua_State* L, int idx);
LuaThread* lua_toluathread(lua_State* L, int idx);
void lua_pushluathread(lua_State* L, LuaThread* s);
void lua_registerluathread(lua_State* L);
