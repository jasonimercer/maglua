#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include "OpenGLScene.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

	OpenGLScene scene;
	lua_State* L;


public slots:
	void fullscreen(bool t);
	void viewmenubar(bool t);
	void tick();
	void run_script();

	void goNext();
	void goPrev();

private:
	Ui::MainWindow *ui;

};

#endif // MAINWINDOW_H
