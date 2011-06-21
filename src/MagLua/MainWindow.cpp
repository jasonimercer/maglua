#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QtGui>

MainWindow::MainWindow(QWidget *parent) :
		QMainWindow(parent),
		ui(new Ui::MainWindow)
{
    ui->setupUi(this);

	connect(ui->mdiArea, SIGNAL(subWindowActivated(QMdiSubWindow*)), this, SLOT(updateMenus()));
	windowMapper = new QSignalMapper(this);
	connect(windowMapper, SIGNAL(mapped(QWidget*)), this, SLOT(setActiveSubWindow(QWidget*)));

	createActions();
	createMenus();
	createToolBars();
	createStatusBar();
	updateMenus();

	readSettings();

	setWindowTitle(tr("MagLua"));
	setUnifiedTitleAndToolBarOnMac(true);

	updateRecentFileActions();
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

void MainWindow::newFile()
{
	DocumentWindow *child = createDocumentChild();
	child->newFile();
	child->show();
}

void MainWindow::open()
{
	QString fileName = QFileDialog::getOpenFileName(this);

	if(!fileName.isEmpty())
	{
		openFile(fileName);
	}
}

void MainWindow::openFile(const QString& fileName)
{
	QMdiSubWindow *existing = findMdiChild(fileName);
	if(existing)
	{
		ui->mdiArea->setActiveSubWindow(existing);
		return;
	}

	DocumentWindow *child = createDocumentChild();
	if(child->loadFile(fileName))
	{
		statusBar()->showMessage(tr("File loaded"), 2000);
//		child->show();
		child->showMaximized();

		QString fileName = activeMdiChild()->currentFile();

		QSettings settings("Mercer", "MagLuaFrontend");
		QStringList files = settings.value("recentFileList").toStringList();
		files.removeAll(fileName);
		files.prepend(fileName);
		while (files.size() > MaxRecentFiles)
			files.removeLast();

		settings.setValue("recentFileList", files);

		updateRecentFileActions();

	}
	else
	{
		child->close();
	}
}

void MainWindow::save()
{
	if(activeMdiChild() && activeMdiChild()->save())
		statusBar()->showMessage(tr("File saved"), 2000);
}

void MainWindow::saveAs()
{
	if(activeMdiChild() && activeMdiChild()->saveAs())
	{
		QString fileName = activeMdiChild()->currentFile();

		QSettings settings("Mercer", "MagLuaFrontend");
		QStringList files = settings.value("recentFileList").toStringList();
		files.removeAll(fileName);
		files.prepend(fileName);
		while (files.size() > MaxRecentFiles)
			files.removeLast();

		settings.setValue("recentFileList", files);

		updateRecentFileActions();

		statusBar()->showMessage(tr("File saved"), 2000);
	}
}

void MainWindow::cut()
{
	if(activeMdiChild())
		activeMdiChild()->cut();
}

void MainWindow::copy()
{
	if(activeMdiChild())
		activeMdiChild()->copy();
}

void MainWindow::paste()
{
	if(activeMdiChild())
		activeMdiChild()->paste();
}

void MainWindow::run()
{
	if(activeMdiChild())
		activeMdiChild()->run();
}

void MainWindow::about()
{
	QMessageBox::about(this, tr("About MagLua"),
		"<center>This program augments the Lua scripting language "
		"with Micromagnetic building blocks.</center>"
		" <p> &copy; 2011 Jason Mercer </p>");
}

void MainWindow::updateMenus()
{
	bool hasMdiChild = (activeMdiChild() != 0);
	saveAct->setEnabled(hasMdiChild);
	saveAsAct->setEnabled(hasMdiChild);
	pasteAct->setEnabled(hasMdiChild);
	closeAct->setEnabled(hasMdiChild);
	closeAllAct->setEnabled(hasMdiChild);
	tileAct->setEnabled(hasMdiChild);
	cascadeAct->setEnabled(hasMdiChild);
	nextAct->setEnabled(hasMdiChild);
	previousAct->setEnabled(hasMdiChild);
	separatorAct->setVisible(hasMdiChild);
	runAct->setEnabled(hasMdiChild);

	bool hasSelection = (activeMdiChild() &&
						 activeMdiChild()->textCursor().hasSelection());
	cutAct->setEnabled(hasSelection);
	copyAct->setEnabled(hasSelection);
}

void MainWindow::updateWindowMenu()
{
	windowMenu->clear();
	windowMenu->addAction(closeAct);
	windowMenu->addAction(closeAllAct);
	windowMenu->addSeparator();
	windowMenu->addAction(tileAct);
	windowMenu->addAction(cascadeAct);
	windowMenu->addSeparator();
	windowMenu->addAction(nextAct);
	windowMenu->addAction(previousAct);
	windowMenu->addAction(separatorAct);

	QList<QMdiSubWindow *> windows = ui->mdiArea->subWindowList();
	separatorAct->setVisible(!windows.isEmpty());

	for (int i = 0; i < windows.size(); ++i)
	{
		DocumentWindow *child = qobject_cast<DocumentWindow*>(windows.at(i)->widget());

		QString text;
		if(i < 9)
		{
			text = tr("&%1 %2").arg(i + 1)
				   .arg(child->userFriendlyCurrentFile());
		}
		else
		{
			text = tr("%1 %2").arg(i + 1)
				   .arg(child->userFriendlyCurrentFile());
		}
		QAction *action  = windowMenu->addAction(text);
		action->setCheckable(true);
		action ->setChecked(child == activeMdiChild());
		connect(action, SIGNAL(triggered()), windowMapper, SLOT(map()));
		windowMapper->setMapping(action, windows.at(i));
	}
}

DocumentWindow* MainWindow::createDocumentChild()
{
	DocumentWindow* child = new DocumentWindow;
	ui->mdiArea->addSubWindow(child);

	connect(child->textEdit(), SIGNAL(copyAvailable(bool)), cutAct,  SLOT(setEnabled(bool)));
	connect(child->textEdit(), SIGNAL(copyAvailable(bool)), copyAct, SLOT(setEnabled(bool)));

	return child;
}

#include "ModuleSelect.h"
void MainWindow::chooseModules()
{
	ModuleSelect ms(this);
	if(ms.exec())
	{
		QSettings settings("Mercer", "MagLuaFrontend");
		settings.setValue("modules", ms.mods);
	}
}

void MainWindow::openRecentFile()
{
	QAction *action = qobject_cast<QAction *>(sender());
	if(action)
	{
		openFile(action->data().toString());
	}
}


void MainWindow::createActions()
{
	newAct = new QAction(QIcon(":/icons/new.png"), tr("&New"), this);
	//newAct = new QAction(style()->standardIcon(QStyle::SP_FileIcon), tr("&New"), this);
	newAct->setShortcuts(QKeySequence::New);
	newAct->setStatusTip(tr("Create a new file"));
	connect(newAct, SIGNAL(triggered()), this, SLOT(newFile()));

	modAct = new QAction(tr("&Choose Modules"), this);
	//newAct = new QAction(style()->standardIcon(QStyle::SP_FileIcon), tr("&New"), this);
	modAct->setShortcut(Qt::ControlModifier + Qt::Key_M);
	modAct->setStatusTip(tr("Choose MagLua Modules"));
	connect(modAct, SIGNAL(triggered()), this, SLOT(chooseModules()));

	openAct = new QAction(QIcon(":/icons/open.png"), tr("&Open..."), this);
	openAct->setShortcuts(QKeySequence::Open);
	openAct->setStatusTip(tr("Open an existing file"));
	connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

	saveAct = new QAction(QIcon(":/icons/save.png"), tr("&Save"), this);
	saveAct->setShortcuts(QKeySequence::Save);
	saveAct->setStatusTip(tr("Save the document to disk"));
	connect(saveAct, SIGNAL(triggered()), this, SLOT(save()));

	saveAsAct = new QAction(tr("Save &As..."), this);
	saveAsAct->setShortcuts(QKeySequence::SaveAs);
	saveAsAct->setStatusTip(tr("Save the document under a new name"));
	connect(saveAsAct, SIGNAL(triggered()), this, SLOT(saveAs()));

	exitAct = new QAction(tr("E&xit"), this);
	exitAct->setShortcuts(QKeySequence::Quit);
	exitAct->setStatusTip(tr("Exit the application"));
	connect(exitAct, SIGNAL(triggered()), qApp, SLOT(closeAllWindows()));

	cutAct = new QAction(QIcon(":/icons/cut.png"), tr("Cu&t"), this);
	cutAct->setShortcuts(QKeySequence::Cut);
	cutAct->setStatusTip(tr("Cut the current selection's contents to the "
							"clipboard"));
	connect(cutAct, SIGNAL(triggered()), this, SLOT(cut()));

	copyAct = new QAction(QIcon(":/icons/copy.png"), tr("&Copy"), this);
	copyAct->setShortcuts(QKeySequence::Copy);
	copyAct->setStatusTip(tr("Copy the current selection's contents to the "
							 "clipboard"));
	connect(copyAct, SIGNAL(triggered()), this, SLOT(copy()));

	pasteAct = new QAction(QIcon(":/icons/paste.png"), tr("&Paste"), this);
	pasteAct->setShortcuts(QKeySequence::Paste);
	pasteAct->setStatusTip(tr("Paste the clipboard's contents into the current "
							  "selection"));
	connect(pasteAct, SIGNAL(triggered()), this, SLOT(paste()));


	runAct = new QAction(QIcon(":/icons/run.png"), tr("&Run"), this);
	runAct->setShortcuts(QKeySequence::Refresh);
	runAct->setStatusTip(tr("Run the current script"));
	connect(runAct, SIGNAL(triggered()), this, SLOT(run()));

	closeAct = new QAction(tr("Cl&ose"), this);
	closeAct->setStatusTip(tr("Close the active window"));
	connect(closeAct, SIGNAL(triggered()),
			ui->mdiArea, SLOT(closeActiveSubWindow()));

	closeAllAct = new QAction(tr("Close &All"), this);
	closeAllAct->setStatusTip(tr("Close all the windows"));
	connect(closeAllAct, SIGNAL(triggered()),
			ui->mdiArea, SLOT(closeAllSubWindows()));

	tileAct = new QAction(tr("&Tile"), this);
	tileAct->setStatusTip(tr("Tile the windows"));
	connect(tileAct, SIGNAL(triggered()), ui->mdiArea, SLOT(tileSubWindows()));

	cascadeAct = new QAction(tr("&Cascade"), this);
	cascadeAct->setStatusTip(tr("Cascade the windows"));
	connect(cascadeAct, SIGNAL(triggered()), ui->mdiArea, SLOT(cascadeSubWindows()));

	nextAct = new QAction(tr("Ne&xt"), this);
	nextAct->setShortcuts(QKeySequence::NextChild);
	nextAct->setStatusTip(tr("Move the focus to the next window"));
	connect(nextAct, SIGNAL(triggered()),
			ui->mdiArea, SLOT(activateNextSubWindow()));

	previousAct = new QAction(tr("Pre&vious"), this);
	previousAct->setShortcuts(QKeySequence::PreviousChild);
	previousAct->setStatusTip(tr("Move the focus to the previous "
								 "window"));
	connect(previousAct, SIGNAL(triggered()),
			ui->mdiArea, SLOT(activatePreviousSubWindow()));

	separatorAct = new QAction(this);
	separatorAct->setSeparator(true);

	aboutAct = new QAction(tr("&About"), this);
	aboutAct->setStatusTip(tr("Show the application's About box"));
	connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

	aboutQtAct = new QAction(tr("About &Qt"), this);
	aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
	connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}

void MainWindow::updateRecentFileActions()
{
	QSettings settings("Mercer", "MagLuaFrontend");
	QStringList files = settings.value("recentFileList").toStringList();

	int numRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

	for (int i = 0; i < numRecentFiles; ++i)
	{
		QString text = tr("&%1 %2").arg(i + 1).arg(strippedName(files[i]));
		recentFileActs[i]->setText(text);
		recentFileActs[i]->setData(files[i]);
		recentFileActs[i]->setVisible(true);
	}

	for (int j = numRecentFiles; j < MaxRecentFiles; ++j)
	{
		recentFileActs[j]->setVisible(false);
	}

//	separatorAct->setVisible(numRecentFiles > 0);
}

QString MainWindow::strippedName(const QString &fullFileName)
{
	return QFileInfo(fullFileName).fileName();
}

void MainWindow::createMenus()
{
	fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(newAct);
	fileMenu->addAction(openAct);
	fileMenu->addAction(saveAct);
	fileMenu->addAction(saveAsAct);
	fileMenu->addSeparator();


	for (int i = 0; i < MaxRecentFiles; ++i)
	{
		recentFileActs[i] = new QAction(this);
		recentFileActs[i]->setVisible(false);
		connect(recentFileActs[i], SIGNAL(triggered()),
				this, SLOT(openRecentFile()));
		fileMenu->addAction(recentFileActs[i]);
	}
	fileMenu->addSeparator();
	fileMenu->addAction(exitAct);


	editMenu = menuBar()->addMenu(tr("&Edit"));
	editMenu->addAction(cutAct);
	editMenu->addAction(copyAct);
	editMenu->addAction(pasteAct);
	editMenu->addSeparator();
	editMenu->addAction(modAct);
	editMenu->addSeparator();
	editMenu->addAction(runAct);

	windowMenu = menuBar()->addMenu(tr("&Window"));
	updateWindowMenu();
	connect(windowMenu, SIGNAL(aboutToShow()), this, SLOT(updateWindowMenu()));

	menuBar()->addSeparator();

	helpMenu = menuBar()->addMenu(tr("&Help"));
	helpMenu->addAction(aboutAct);
	helpMenu->addAction(aboutQtAct);
}

void MainWindow::createToolBars()
{
	fileToolBar = addToolBar(tr("File"));
	fileToolBar->addAction(newAct);
	fileToolBar->addAction(openAct);
	fileToolBar->addAction(saveAct);

	editToolBar = addToolBar(tr("Edit"));
	editToolBar->addAction(cutAct);
	editToolBar->addAction(copyAct);
	editToolBar->addAction(pasteAct);
	editToolBar->addSeparator();
	editToolBar->addAction(runAct);

}

void MainWindow::createStatusBar()
{
	statusBar()->showMessage(tr("Ready"));
}

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
