#include "Classes.h"

#include <QFileDialog>
#include "MainWindow.h"
#include "QLineNumberTextEdit.h"
#include "QLuaSplitter.h"
#include "QLuaTabWidget.h"
#include "QLuaLabel.h"
#include "ChildWindow.h"
#include "QLuaPushButton.h"
#include "QLuaLayout.h"
#include "QLuaLineEdit.h"
#include "QLuaSettings.h"
#include "QLuaToolbar.h"
#include "QLuaAction.h"
#include "QLuaTimer.h"
#include "QLuaMenu.h"

#include "LuaThread.h"

int lua_isluawidget(lua_State* L, int idx)
{
	return lua_toluawidget(L, idx) != 0;
}


QLuaWidget* lua_toluawidget(lua_State* L, int idx)
{

	if(lua_islualineedit(L, idx))
	{
		return lua_tolualineedit(L, idx);
	}

	if(lua_istoolbar(L, idx))
	{
		return lua_toluatoolbar(L, idx);
	}

	if(lua_ismenu(L, idx))
	{
		return lua_toluamenu(L, idx);
	}

	if(lua_issplitter(L, idx))
	{
		return lua_tosplitter(L, idx);
	}

	if(lua_istabwidget(L, idx))
	{
		return lua_totabwidget(L, idx);
	}

	if(lua_islabel(L, idx))
	{
		return lua_tolualabel(L, idx);
	}
	if(lua_istextedit(L, idx))
	{
		return lua_toluatextedit(L, idx);
	}
	if(lua_ispushbutton(L, idx))
	{
		return lua_topushbutton(L, idx);
	}

	if(lua_islayout(L, idx))
	{
		return lua_tolayout(L, idx);
	}

	return 0;
}


QWidget* lua_towidget(lua_State* L, int idx)
{
	QLuaWidget* ww = lua_toluawidget(L, idx);
	if(ww)
		return ww->widget;
	return 0;
}

static int l_setfocus(lua_State* L)
{
	QLuaWidget* w = lua_toluawidget(L, 1);
	if(!w) return 0;

	w->setFocus();
	return 0;
}

static int l_savefilename(lua_State* L)
{
	QString curFile;

	if(lua_isstring(L, -1))
		curFile = lua_tostring(L, -1);

	QString fileName = QFileDialog::getSaveFileName(Singleton.mainWindow, "Save As",
													curFile);
	if(fileName.length())
	{
		lua_pushstring(L, fileName.toStdString().c_str());
		return 1;
	}
	return 0;
}

static int l_openfilenames(lua_State* L)
{
	QString t = lua_tostring(L, 1);
	QStringList files = QFileDialog::getOpenFileNames(
							Singleton.mainWindow,
							"Select one or more files to open",
							"",
							t);

	lua_newtable(L);
	for(int i=0; i<files.size(); i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, files.at(i).toStdString().c_str());
		lua_settable(L, -3);
	}
	return 1;
}
static int l_openfilename(lua_State* L)
{
	QString t = lua_tostring(L, 1);
	QString file = QFileDialog::getOpenFileName(
							Singleton.mainWindow,
							"Select a files to open",
							"",
							t);

	if(file.length())
		lua_pushstring(L, file.toStdString().c_str());
	else
		lua_pushnil(L);
	return 1;
}


static int l_newmenu(lua_State* L)
{
	QString name = lua_tostring(L, 1);
	if(name.length() == 0) return 0;


	QMenu* m = Singleton.mainWindow->menuBar()->addMenu(name);
	lua_pushmenu(L, m);
	return 1;
}

static int l_appShutdown(lua_State* /* L */)
{
	Singleton.mainWindow->quit();

	return 0;
}


int lua_registerwidgets(lua_State* L)
{
	lua_registerwindow(L);
	lua_registertextedit(L);
	lua_registersplitter(L);
	lua_registertabwidget(L);
	lua_registerlabel(L);
	lua_registerpushbutton(L);
	lua_registerlayout(L);
	lua_registerlineedit(L);
	lua_registerqluasettings(L);
	lua_registertoolbar(L);
	lua_registeraction(L);
	lua_registermenu(L);
	lua_registertimer(L);

	lua_pushcfunction(L, l_savefilename);
	lua_setglobal(L, "getSaveFileName");
	lua_pushcfunction(L, l_openfilenames);
	lua_setglobal(L, "getOpenFileNames");
	lua_pushcfunction(L, l_openfilename);
	lua_setglobal(L, "getOpenFileName");

	lua_pushcfunction(L, l_newmenu);
	lua_setglobal(L, "newMenu");

	lua_pushcfunction(L, l_setfocus);
	lua_setglobal(L, "setFocus");

	lua_pushcfunction(L, l_appShutdown);
	lua_setglobal(L, "appQuit");

	lua_registerluathread(L);
	return 0;
}
