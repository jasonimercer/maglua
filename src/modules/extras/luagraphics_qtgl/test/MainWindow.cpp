#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "libMagLua.h"
#include <QGLWidget>
#include <QTimer>
#include <QSettings>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	setStatusBar(0);
//	menuBar()->setVisible(false);

    L = lua_open();
	scene.registerFunctions(L);
	scene.setSceneRect(-100000, -100000, 200000, 200000);
	scene.setBackgroundBrush(Qt::white);

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
	ui->view->registerFunctions(L);
	//ui->view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	//ui->view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);


	libMagLua(L, 1, 1);
	lua_pushlightuserdata(L, &scene);
	lua_setglobal(L, "scene_userdata");

	if(luaL_dofile(L, "startup.lua"))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
	}

	QSettings settings("Mercer", "MagLuaGUI");

	//restore window props from last close
	settings.beginGroup("MainWindow");
	resize(settings.value("size", QSize(600, 400)).toSize());
	move(settings.value("pos", QPoint(200, 200)).toPoint());
	restoreState(settings.value("state").toByteArray());
	settings.endGroup();


	QTimer* t = new QTimer(this);
	t->setInterval(1000.0/24.0);
	connect(t, SIGNAL(timeout()), this, SLOT(tick()));
	t->start();

}

void MainWindow::tick()
{
	//ui->view->update();
	scene.update();
}


MainWindow::~MainWindow()
{
	//lua_close(L);

	QSettings settings("Mercer", "MagLuaGUI");
	settings.beginGroup("MainWindow");
	settings.setValue("size", size());
	settings.setValue("pos", pos());
	settings.setValue("state", saveState());
	settings.endGroup();


    delete ui;
}
