#ifndef QLUATIMER_H
#define QLUATIMER_H

#include "QLuaWidget.h"
#include <QObject>

class QLuaTimer : public QObject
{
    Q_OBJECT
public:
	explicit QLuaTimer(lua_State* L);
	~QLuaTimer();

	void setFunction(int ref);
	void oneshot(int ms);
	void start(int ms);
	void stop();
	bool running();

	int refcount;
	int gcd;
	QTimer* timer;

public slots:
	void timeout();

private:
	int funcref;
	lua_State* L;
};

#endif // QLUATIMER_H


void lua_registertimer(lua_State* L);
void lua_pushluatimer(lua_State* L, QLuaTimer* c);
QTimer* lua_totimer(lua_State* L, int idx);
QLuaTimer* lua_toluatimer(lua_State* L, int idx);
int lua_istimer(lua_State* L, int idx);
