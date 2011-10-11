#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#define ORGANIZATION "Mercer"
#define APPLICATION "MagLuaFrontend"
#define AUTHOR_COPYRIGHT_BANNER "MagLua by Jason Mercer (c) 2011"

#include <QMainWindow>
#include <QSignalMapper>
#include "DocumentWindow.h"
#include <QMdiSubWindow>
#include "ChildWindow.h"

#include <iostream>
using namespace std;

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

	Ui::MainWindow *ui;
	lua_State* L;

	void quit();

protected:
	void closeEvent(QCloseEvent *event);

public slots:
	void threadedCallWithMsg(lua_State* L, int ref, const QString& msg);


private slots:
//	void newFile();
//	void open();
//	void openFile(const QString& filename);
//	void save();
//	void saveAs();
//	void cut();
//	void run();
//	void stop();
//	void copy();
//	void paste();
//	void find();
//	void about();
//	void updateMenus();
//	void chooseModules();
//	void updateWindowMenu();
	DocumentWindow* createDocumentChild();
	void setActiveSubWindow(QWidget *window);
//	void openRecentFile();



private:
//	void createActions();
//	void createMenus();
//	void createToolBars();
//	void createStatusBar();
	void readSettings();
	void writeSettings();
//	void updateRecentFileActions();

	void showLuaError(QString s);

	QString strippedName(const QString &fullFileName);

	DocumentWindow* activeMdiChild();
	ChildWindow* activeLuaWindow();

	QMdiSubWindow* findMdiChild(const QString &fileName);


	QSignalMapper *windowMapper;

	QMenu *fileMenu;
	QMenu *editMenu;
	QMenu *windowMenu;
	QMenu *helpMenu;
	QToolBar *fileToolBar;
	QToolBar *editToolBar;
	QAction *modAct;
	QAction *newAct;
	QAction *openAct;
	QAction *saveAct;
	QAction *saveAsAct;
	QAction *exitAct;
	QAction *cutAct;
	QAction *copyAct;
	QAction *runAct;
	QAction *stopAct;
	QAction *pasteAct;
	QAction *findAct;
	QAction *closeAct;
	QAction *closeAllAct;
	QAction *tileAct;
	QAction *cascadeAct;
	QAction *nextAct;
	QAction *previousAct;
	QAction *separatorAct;
	QAction *aboutAct;
	QAction *aboutQtAct;

	enum { MaxRecentFiles = 5 };
	QAction *recentFileActs[MaxRecentFiles];
};

#define Singleton (_Singleton::instance())
class _Singleton
{
public:
	static _Singleton& instance()
	{
		if(ptr == 0)
			ptr = new _Singleton;
		return *ptr;
	}

	static MainWindow* mainWindow;
	static float time;
protected:
	_Singleton() {}

private:
	static _Singleton* ptr;
};

#endif // MAINWINDOW_H
