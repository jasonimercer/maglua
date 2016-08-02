#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QtGui>

#include <iostream>
using namespace std;

#include "Classes.h"

_Singleton* _Singleton::ptr = 0;
MainWindow* _Singleton::mainWindow = 0;
float       _Singleton::time = 0;

MainWindow::MainWindow(QWidget *parent) :
		QMainWindow(parent),
		ui(new Ui::MainWindow)
{
	Singleton.mainWindow = this;

    ui->setupUi(this);

	L = lua_open();
	luaL_openlibs(L);
	lua_registerwidgets(L);

	readSettings();

	setWindowTitle(tr("MagLua"));
	setUnifiedTitleAndToolBarOnMac(true);

	if(luaL_dofile(L, "test_ui.lua"))
		cout << lua_tostring(L, -1) << endl;

}

void MainWindow::quit()
{
	// do any pre-shutdown housecleaning
	qApp->closeAllWindows();
}

void MainWindow::threadedCallWithMsg(lua_State* LL, int ref, const QString& msg)
{
	if(ref != LUA_NOREF)
	{
		lua_rawgeti(LL, LUA_REGISTRYINDEX, ref);
		lua_pushstring(LL, msg.toStdString().c_str());

		if(lua_pcall(LL, 1, 0, 0))
		{
			cerr << lua_tostring(LL, 1) << endl;
			lua_pop(LL, 1);
		}
	}
}


MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
	ui->mdiArea->closeAllSubWindows();
	if(ui->mdiArea->currentSubWindow())
	{
		event->ignore();
	}
	else
	{
		writeSettings();
		event->accept();
	}
}



//void MainWindow::cut()
//{
//	if(activeMdiChild())
//		activeMdiChild()->cut();
//}

//void MainWindow::copy()
//{
//	if(activeMdiChild())
//		activeMdiChild()->copy();
//}

//void MainWindow::paste()
//{
//	if(activeMdiChild())
//		activeMdiChild()->paste();
//}

void MainWindow::showLuaError(QString s)
{
	QErrorMessage em(this);

	em.showMessage(s);
	em.exec();

	cerr << s.toStdString() << endl;
}



//void MainWindow::about()
//{
//	QMessageBox::about(this, tr("About MagLua"),
//		"<center>This program augments the Lua scripting language "
//		"with Micromagnetic building blocks.</center>"
//		" <p> &copy; 2011 Jason Mercer </p>");
//}

//void MainWindow::updateMenus()
//{
//	bool hasMdiChild = (activeMdiChild() != 0 || activeLuaWindow() != 0);
//	saveAct->setEnabled(hasMdiChild);
//	saveAsAct->setEnabled(hasMdiChild);
//	pasteAct->setEnabled(hasMdiChild);
//	findAct->setEnabled(hasMdiChild);
//	closeAct->setEnabled(hasMdiChild);
//	closeAllAct->setEnabled(hasMdiChild);
//	tileAct->setEnabled(hasMdiChild);
//	cascadeAct->setEnabled(hasMdiChild);
//	nextAct->setEnabled(hasMdiChild);
//	previousAct->setEnabled(hasMdiChild);
//	separatorAct->setVisible(hasMdiChild);
//	runAct->setEnabled(hasMdiChild);
//	stopAct->setEnabled(hasMdiChild);

//	bool hasSelection = (activeMdiChild() &&
//						 activeMdiChild()->textCursor().hasSelection());
//	cutAct->setEnabled(hasSelection);
//	copyAct->setEnabled(hasSelection);
//}

//void MainWindow::updateWindowMenu()
//{
//	windowMenu->clear();
//	windowMenu->addAction(closeAct);
//	windowMenu->addAction(closeAllAct);
//	windowMenu->addSeparator();
//	windowMenu->addAction(tileAct);
//	windowMenu->addAction(cascadeAct);
//	windowMenu->addSeparator();
//	windowMenu->addAction(nextAct);
//	windowMenu->addAction(previousAct);
//	windowMenu->addAction(separatorAct);

//	QList<QMdiSubWindow *> windows = ui->mdiArea->subWindowList();
//	separatorAct->setVisible(!windows.isEmpty());

//	for (int i = 0; i < windows.size(); ++i)
//	{
//		DocumentWindow *child = qobject_cast<DocumentWindow*>(windows.at(i)->widget());

//		QString text;
//		if(child)
//		{
//			if(i < 9)
//			{
//				text = tr("&%1 %2").arg(i + 1)
//					   .arg(child->userFriendlyCurrentFile());
//			}
//			else
//			{
//				text = tr("%1 %2").arg(i + 1)
//					   .arg(child->userFriendlyCurrentFile());
//			}
//			QAction *action  = windowMenu->addAction(text);
//			action->setCheckable(true);
//			action ->setChecked(child == activeMdiChild());
//			connect(action, SIGNAL(triggered()), windowMapper, SLOT(map()));
//			windowMapper->setMapping(action, windows.at(i));
//		}
//	}
//}

DocumentWindow* MainWindow::createDocumentChild()
{
	DocumentWindow* child = new DocumentWindow;
	ui->mdiArea->addSubWindow(child);

	connect(child->textEdit(), SIGNAL(copyAvailable(bool)), cutAct,  SLOT(setEnabled(bool)));
	connect(child->textEdit(), SIGNAL(copyAvailable(bool)), copyAct, SLOT(setEnabled(bool)));

	return child;
}

//void MainWindow::updateRecentFileActions()
//{
//	QSettings settings("Mercer", "MagLuaFrontend");
//	QStringList files = settings.value("recentFileList").toStringList();

//	int numRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

//	for (int i = 0; i < numRecentFiles; ++i)
//	{
//		QString text = tr("&%1 %2").arg(i + 1).arg(strippedName(files[i]));
//		recentFileActs[i]->setText(text);
//		recentFileActs[i]->setData(files[i]);
//		recentFileActs[i]->setVisible(true);
//	}

//	for (int j = numRecentFiles; j < MaxRecentFiles; ++j)
//	{
//		recentFileActs[j]->setVisible(false);
//	}

////	separatorAct->setVisible(numRecentFiles > 0);
//}

QString MainWindow::strippedName(const QString &fullFileName)
{
	return QFileInfo(fullFileName).fileName();
}



//void MainWindow::createStatusBar()
//{
//	statusBar()->showMessage(tr("Ready"));
//}

void MainWindow::readSettings()
{
	QSettings settings("Mercer", "MagLuaFrontend");
	QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
	QSize size = settings.value("size", QSize(400, 400)).toSize();
	move(pos);
	resize(size);
}

void MainWindow::writeSettings()
{
	QSettings settings("Mercer", "MagLuaFrontend");
	settings.setValue("pos", pos());
	settings.setValue("size", size());
}

DocumentWindow* MainWindow::activeMdiChild()
{
	if(QMdiSubWindow *activeSubWindow = ui->mdiArea->activeSubWindow())
		return qobject_cast<DocumentWindow*>(activeSubWindow->widget());
	return 0;
}
ChildWindow* MainWindow::activeLuaWindow()
{
	if(QMdiSubWindow *activeSubWindow = ui->mdiArea->activeSubWindow())
		return qobject_cast<ChildWindow*>(activeSubWindow->widget());
	return 0;
}

QMdiSubWindow *MainWindow::findMdiChild(const QString &fileName)
{
	QString canonicalFilePath = QFileInfo(fileName).canonicalFilePath();

	foreach (QMdiSubWindow *window, ui->mdiArea->subWindowList())
	{
		DocumentWindow* mdiChild = qobject_cast<DocumentWindow*>(window->widget());
		if(mdiChild->currentFile() == canonicalFilePath)
			return window;
	}
	return 0;
}

void MainWindow::setActiveSubWindow(QWidget *window)
{
	if(!window)
		return;
	ui->mdiArea->setActiveSubWindow(qobject_cast<QMdiSubWindow *>(window));
}
