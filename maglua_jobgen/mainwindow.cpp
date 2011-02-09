#include <QtGui>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "jobwidget.h"

#define TIMEOUT 2000

MainWindow::MainWindow(QWidget *parent) :
		QMainWindow(parent),
		ui(new Ui::MainWindow)
{
    ui->setupUi(this);

	ui->mdiArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	ui->mdiArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	windowMapper = new QSignalMapper(this);
	connect(windowMapper, SIGNAL(mapped(QWidget*)), this, SLOT(setActiveSubWindow(QWidget*)));

	connect(ui->action_New, SIGNAL(triggered()), this, SLOT(newJob()));


	createActions();
	createMenus();
	createToolBars();
	createStatusBar();
	updateMenus();

	readSettings();

	setUnifiedTitleAndToolBarOnMac(true);

	JobWidget* child = createMdiChild();
	child->showMaximized();
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

void MainWindow::newJob()
{
	JobWidget* child = createMdiChild();

//	child->newFile();
	child->show();
}

void MainWindow::open()
{
	QString fileName = QFileDialog::getOpenFileName(this);

	if(!fileName.isEmpty())
	{
		QMdiSubWindow *existing = findMdiChild(fileName);
		if (existing)
		{
			ui->mdiArea->setActiveSubWindow(existing);
			return;
		}

		JobWidget *child = createMdiChild();

		if(child->loadFile(fileName))
		{
			statusBar()->showMessage(tr("File loaded"), TIMEOUT);
			child->show();
		}
		else
		{
			child->close();
		}
	}
}

void MainWindow::save()
{
	if(activeMdiChild() && activeMdiChild()->save())
	{
		statusBar()->showMessage(tr("File saved"), TIMEOUT);
	}
}

void MainWindow::saveAs()
{
	if(activeMdiChild() && activeMdiChild()->saveAs())
	{
		statusBar()->showMessage(tr("File saved"), TIMEOUT);
	}
}

//void MainWindow::cut()
//{
//	if(activeMdiChild())
//		activeMdiChild()->cut();
//}

//void MainWindow::copy()
//{
//	if (activeMdiChild())
//		activeMdiChild()->copy();
//}

//void MainWindow::paste()
//{
//	if (activeMdiChild())
//		activeMdiChild()->paste();
//}

void MainWindow::updateMenus()
{
//	bool hasMdiChild = (activeMdiChild() != 0);
//	saveAct->setEnabled(hasMdiChild);
//	saveAsAct->setEnabled(hasMdiChild);
//	pasteAct->setEnabled(hasMdiChild);
//	closeAct->setEnabled(hasMdiChild);
//	closeAllAct->setEnabled(hasMdiChild);
//	tileAct->setEnabled(hasMdiChild);
//	cascadeAct->setEnabled(hasMdiChild);
//	nextAct->setEnabled(hasMdiChild);
//	previousAct->setEnabled(hasMdiChild);
//	separatorAct->setVisible(hasMdiChild);

//	bool hasSelection = (activeMdiChild() &&
//						 activeMdiChild()->textCursor().hasSelection());
//	cutAct->setEnabled(hasSelection);
//	copyAct->setEnabled(hasSelection);
}

void MainWindow::updateWindowMenu()
{
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

//	QList<QMdiSubWindow *> windows = mdiArea->subWindowList();
//	separatorAct->setVisible(!windows.isEmpty());

//	for (int i = 0; i < windows.size(); ++i) {
//		MdiChild *child = qobject_cast<MdiChild *>(windows.at(i)->widget());

//		QString text;
//		if (i < 9) {
//			text = tr("&%1 %2").arg(i + 1)
//				   .arg(child->userFriendlyCurrentFile());
//		} else {
//			text = tr("%1 %2").arg(i + 1)
//				   .arg(child->userFriendlyCurrentFile());
//		}
//		QAction *action  = windowMenu->addAction(text);
//		action->setCheckable(true);
//		action ->setChecked(child == activeMdiChild());
//		connect(action, SIGNAL(triggered()), windowMapper, SLOT(map()));
//		windowMapper->setMapping(action, windows.at(i));
//	}
}

JobWidget *MainWindow::createMdiChild()
{
	JobWidget* child = new JobWidget;

	ui->mdiArea->addSubWindow(child);

//	connect(child, SIGNAL(copyAvailable(bool)),
//			cutAct, SLOT(setEnabled(bool)));
//	connect(child, SIGNAL(copyAvailable(bool)),
//			copyAct, SLOT(setEnabled(bool)));

	return child;
}

void MainWindow::createActions()
{
//	newAct = new QAction(QIcon(":/images/new.png"), tr("&New"), this);
//	newAct->setShortcuts(QKeySequence::New);
//	newAct->setStatusTip(tr("Create a new file"));
//	connect(newAct, SIGNAL(triggered()), this, SLOT(newFile()));

//	openAct = new QAction(QIcon(":/images/open.png"), tr("&Open..."), this);
//	openAct->setShortcuts(QKeySequence::Open);
//	openAct->setStatusTip(tr("Open an existing file"));
//	connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

//	saveAct = new QAction(QIcon(":/images/save.png"), tr("&Save"), this);
//	saveAct->setShortcuts(QKeySequence::Save);
//	saveAct->setStatusTip(tr("Save the document to disk"));
//	connect(saveAct, SIGNAL(triggered()), this, SLOT(save()));

//	saveAsAct = new QAction(tr("Save &As..."), this);
//	saveAsAct->setShortcuts(QKeySequence::SaveAs);
//	saveAsAct->setStatusTip(tr("Save the document under a new name"));
//	connect(saveAsAct, SIGNAL(triggered()), this, SLOT(saveAs()));

//	exitAct = new QAction(tr("E&xit"), this);
//	exitAct->setShortcuts(QKeySequence::Quit);
//	exitAct->setStatusTip(tr("Exit the application"));
//	connect(exitAct, SIGNAL(triggered()), qApp, SLOT(closeAllWindows()));

//	cutAct = new QAction(QIcon(":/images/cut.png"), tr("Cu&t"), this);
//	cutAct->setShortcuts(QKeySequence::Cut);
//	cutAct->setStatusTip(tr("Cut the current selection's contents to the "
//							"clipboard"));
//	connect(cutAct, SIGNAL(triggered()), this, SLOT(cut()));

//	copyAct = new QAction(QIcon(":/images/copy.png"), tr("&Copy"), this);
//	copyAct->setShortcuts(QKeySequence::Copy);
//	copyAct->setStatusTip(tr("Copy the current selection's contents to the "
//							 "clipboard"));
//	connect(copyAct, SIGNAL(triggered()), this, SLOT(copy()));

//	pasteAct = new QAction(QIcon(":/images/paste.png"), tr("&Paste"), this);
//	pasteAct->setShortcuts(QKeySequence::Paste);
//	pasteAct->setStatusTip(tr("Paste the clipboard's contents into the current "
//							  "selection"));
//	connect(pasteAct, SIGNAL(triggered()), this, SLOT(paste()));

//	closeAct = new QAction(tr("Cl&ose"), this);
//	closeAct->setStatusTip(tr("Close the active window"));
//	connect(closeAct, SIGNAL(triggered()),
//			mdiArea, SLOT(closeActiveSubWindow()));

//	closeAllAct = new QAction(tr("Close &All"), this);
//	closeAllAct->setStatusTip(tr("Close all the windows"));
//	connect(closeAllAct, SIGNAL(triggered()),
//			mdiArea, SLOT(closeAllSubWindows()));

//	tileAct = new QAction(tr("&Tile"), this);
//	tileAct->setStatusTip(tr("Tile the windows"));
//	connect(tileAct, SIGNAL(triggered()), mdiArea, SLOT(tileSubWindows()));

//	cascadeAct = new QAction(tr("&Cascade"), this);
//	cascadeAct->setStatusTip(tr("Cascade the windows"));
//	connect(cascadeAct, SIGNAL(triggered()), mdiArea, SLOT(cascadeSubWindows()));

//	nextAct = new QAction(tr("Ne&xt"), this);
//	nextAct->setShortcuts(QKeySequence::NextChild);
//	nextAct->setStatusTip(tr("Move the focus to the next window"));
//	connect(nextAct, SIGNAL(triggered()),
//			mdiArea, SLOT(activateNextSubWindow()));

//	previousAct = new QAction(tr("Pre&vious"), this);
//	previousAct->setShortcuts(QKeySequence::PreviousChild);
//	previousAct->setStatusTip(tr("Move the focus to the previous "
//								 "window"));
//	connect(previousAct, SIGNAL(triggered()),
//			mdiArea, SLOT(activatePreviousSubWindow()));

//	separatorAct = new QAction(this);
//	separatorAct->setSeparator(true);

//	aboutAct = new QAction(tr("&About"), this);
//	aboutAct->setStatusTip(tr("Show the application's About box"));
//	connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

//	aboutQtAct = new QAction(tr("About &Qt"), this);
//	aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
//	connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}

void MainWindow::createMenus()
{
//	fileMenu = menuBar()->addMenu(tr("&File"));
//	fileMenu->addAction(newAct);
//	fileMenu->addAction(openAct);
//	fileMenu->addAction(saveAct);
//	fileMenu->addAction(saveAsAct);
//	fileMenu->addSeparator();
//	QAction *action = fileMenu->addAction(tr("Switch layout direction"));
//	connect(action, SIGNAL(triggered()), this, SLOT(switchLayoutDirection()));
//	fileMenu->addAction(exitAct);

//	editMenu = menuBar()->addMenu(tr("&Edit"));
//	editMenu->addAction(cutAct);
//	editMenu->addAction(copyAct);
//	editMenu->addAction(pasteAct);

//	windowMenu = menuBar()->addMenu(tr("&Window"));
//	updateWindowMenu();
//	connect(windowMenu, SIGNAL(aboutToShow()), this, SLOT(updateWindowMenu()));

//	menuBar()->addSeparator();

//	helpMenu = menuBar()->addMenu(tr("&Help"));
//	helpMenu->addAction(aboutAct);
//	helpMenu->addAction(aboutQtAct);
}

void MainWindow::createToolBars()
{
//	fileToolBar = addToolBar(tr("File"));
//	fileToolBar->addAction(newAct);
//	fileToolBar->addAction(openAct);
//	fileToolBar->addAction(saveAct);

//	editToolBar = addToolBar(tr("Edit"));
//	editToolBar->addAction(cutAct);
//	editToolBar->addAction(copyAct);
//	editToolBar->addAction(pasteAct);
}

void MainWindow::createStatusBar()
{
	statusBar()->showMessage("Ready");
}

void MainWindow::readSettings()
{
	QSettings settings("Mercer", "MagLuaJobGen");
	QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
	QSize size = settings.value("size", QSize(400, 400)).toSize();
	move(pos);
	resize(size);
}

void MainWindow::writeSettings()
{
	QSettings settings("Mercer", "MagLuaJobGen");
	settings.setValue("pos", pos());
	settings.setValue("size", size());
}

JobWidget *MainWindow::activeMdiChild()
{
	if (QMdiSubWindow *activeSubWindow = ui->mdiArea->activeSubWindow())
		return qobject_cast<JobWidget *>(activeSubWindow->widget());
	return 0;
}

QMdiSubWindow *MainWindow::findMdiChild(const QString &fileName)
{
	QString canonicalFilePath = QFileInfo(fileName).canonicalFilePath();

	foreach(QMdiSubWindow *window, ui->mdiArea->subWindowList())
	{
		JobWidget *mdiChild = qobject_cast<JobWidget *>(window->widget());
		if (mdiChild->currentFile() == canonicalFilePath)
			return window;
	}
	return 0;
}

void MainWindow::switchLayoutDirection()
{
	if (layoutDirection() == Qt::LeftToRight)
		qApp->setLayoutDirection(Qt::RightToLeft);
	else
		qApp->setLayoutDirection(Qt::LeftToRight);
}

void MainWindow::setActiveSubWindow(QWidget *window)
{
	if (!window)
		return;
	ui->mdiArea->setActiveSubWindow(qobject_cast<QMdiSubWindow *>(window));
}
