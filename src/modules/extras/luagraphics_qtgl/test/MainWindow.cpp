#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "libMagLua.h"
#include <QGLWidget>
#include <QTimer>
#include <QSettings>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	setStatusBar(0);
//	menuBar()->setVisible(false);

    L = lua_open();

	scene.registerFunctions(L);
	scene.setSceneRect(0, 0, 200000, 200000);
	scene.setBackgroundBrush(Qt::white);

	ui->view->setScene(&scene);

	ui->view->setRenderHints(    QPainter::Antialiasing
							   | QPainter::SmoothPixmapTransform
							   | QPainter::TextAntialiasing
							   | QPainter::HighQualityAntialiasing);
	ui->view->setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers)));
	ui->view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	ui->view->setScene(&scene);
	ui->view->setRenderHint(QPainter::Antialiasing, true);
	ui->view->setRenderHint(QPainter::SmoothPixmapTransform, true);
	ui->view->registerFunctions(L);
	ui->view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	ui->view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	//add the actions of the menus to the application so they are available when the menu is hidden
	QList<QAction*> al = ui->menuBar->actions();
	for(int i=0; i<al.size(); i++)
	{
		QAction* a = al.at(i);
		addAction(a);
	}

	ui->action_Fullscreen->setChecked(isFullScreen());
	ui->action_Menubar->setChecked(1);

	connect(ui->action_Fullscreen,  SIGNAL(triggered(bool)), this, SLOT(fullscreen(bool)));
	connect(ui->action_Menubar,    SIGNAL(triggered(bool)), this, SLOT(viewmenubar(bool)));

	connect(ui->action_Next, SIGNAL(triggered()), this, SLOT(goNext()));
	connect(ui->action_Previous, SIGNAL(triggered()), this, SLOT(goPrev()));

	libMagLua(L, 1, 1);
	lua_pushlightuserdata(L, &scene);
	lua_setglobal(L, "scene_userdata");


	QSettings settings("Mercer", "MagLuaGUI");

	//restore window props from last close
	settings.beginGroup("MainWindow");
	resize(settings.value("size", QSize(600, 400)).toSize());
	move(settings.value("pos", QPoint(200, 200)).toPoint());
	restoreState(settings.value("state").toByteArray());
	settings.endGroup();

	connect(ui->action_Run_Script, SIGNAL(triggered()), this, SLOT(run_script()));

	QTimer* t = new QTimer(this);
	t->setInterval(1000.0/24.0);
	connect(t, SIGNAL(timeout()), this, SLOT(tick()));
	t->start();

}

void MainWindow::run_script()
{
	QSettings settings("Mercer", "MagLuaGUI");

	QString lastFile = settings.value("lastFile", "").toString();

	QString filename =
		QFileDialog::getOpenFileName(this,
        tr("Open Script"), lastFile, tr("Script Files (*.lua)"));

	if(filename.size())
	{
		settings.setValue("lastFile", filename);

		if(luaL_dofile(L, filename.toStdString().c_str()))
		{
			fprintf(stderr, "%s\n", lua_tostring(L, -1));
		}

	}

}


void MainWindow::tick()
{
	//ui->view->update();
	scene.update();
}

void MainWindow::goNext()
{
	if(!L) return;
	lua_getglobal(L, "next");
	if(lua_isfunction(L, -1))
	{
		if(lua_pcall(L, 0, 0,0))
		{
			fprintf(stderr, "%s\n", lua_tostring(L, -1));
		}
	}
	else
		lua_pop(L, 1);
}

void MainWindow::goPrev()
{
	if(!L) return;
	lua_getglobal(L, "prev");
	if(lua_isfunction(L, -1))
	{
		if(lua_pcall(L, 0, 0,0))
		{
			fprintf(stderr, "%s\n", lua_tostring(L, -1));
		}
	}
	else
		lua_pop(L, 1);
}

void MainWindow::fullscreen(bool t)
{
	if(t)
		showFullScreen();
	else
		showNormal();
}

void MainWindow::viewmenubar(bool t)
{
	if(t)
		ui->menuBar->show();
	else
		ui->menuBar->hide();
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
