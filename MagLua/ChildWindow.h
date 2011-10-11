#ifndef CHILDWINDOW_H
#define CHILDWINDOW_H

#include <QMainWindow>
#include <QLuaWidget.h>
class QMdiSubWindow;
namespace Ui {
    class ChildWindow;
}

class ChildWindow : public QMainWindow
{
    Q_OBJECT

public:
	explicit ChildWindow(lua_State* L, QWidget *parent = 0);
    ~ChildWindow();

	int lua_add(lua_State* L, int idx);
	void setIOFunction(int f);

	int refcount;
	int iofuncref;

	int callIOFunction(int nargs);

	lua_State* L;
	QMdiSubWindow* subProxy;
private:
	QList<QLuaWidget*> children;
    Ui::ChildWindow *ui;

};

int lua_iswindow(lua_State* L, int idx);
ChildWindow* lua_towindow(lua_State* L, int idx);
void lua_pushwindow(lua_State* L, ChildWindow* c);
void lua_registerwindow(lua_State* L);




#endif // CHILDWINDOW_H
