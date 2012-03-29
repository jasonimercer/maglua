#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "libMagLua.h"
#include <QGLWidget>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	setStatusBar(0);
//	menuBar()->setVisible(false);

	ui->view->setScene(&scene);

	ui->view->setRenderHints(  QPainter::Antialiasing
							   | QPainter::SmoothPixmapTransform
							   | QPainter::TextAntialiasing
							   | QPainter::HighQualityAntialiasing);
	ui->view->setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers)));
	ui->view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	ui->view->setScene(&scene);
	ui->view->setRenderHint(QPainter::Antialiasing, true);
	ui->view->setRenderHint(QPainter::SmoothPixmapTransform, true);

	ui->view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	ui->view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);


    L = lua_open();
	scene.L = L;

    libMagLua(L, 1, 1);
	lua_pushlightuserdata(L, &scene);
	lua_setglobal(L, "scene_userdata");

	if(luaL_dofile(L, "startup.lua"))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
	}
}

MainWindow::~MainWindow()
{
	//lua_close(L);
    delete ui;
}
