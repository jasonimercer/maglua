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
	bool stopRequested() const;
	bool running() const {return L!=0;}
	void setCurrentLine(int line, const QString& src);

signals:
	void printOutput(const QString& text);
	void printError(const QString& text);
	void currentLineChange(int line, const QString& src);


protected:
	void run();

private:
	lua_State* L;
	QMutex mutex;
	QWaitCondition condition;
	bool requestStop;

	int currentLine;
	QString currentSource;
};

#endif // LUATHREAD_H
