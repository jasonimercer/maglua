#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include <QObject>
extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}

class SignalSink : public QObject
{
    Q_OBJECT
public:
	explicit SignalSink(lua_State* L = 0, int funcref = LUA_REFNIL, QObject *parent = 0);
	~SignalSink();

	lua_State* L;
signals:

public slots:
	void activate();
	void activateInt(int i);

private:
	int ref;
};

#endif // SIGNALSINK_H
