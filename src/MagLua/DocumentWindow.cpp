#include "DocumentWindow.h"
#include "ui_DocumentWindow.h"
#include <QtGui>
#include <QFile>
#include <iostream>
#include "LoadLibs.h"
#include <QSettings>
#include <QErrorMessage>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

DocumentWindow::DocumentWindow(QWidget *parent) :
		QMainWindow(parent),
		ui(new Ui::DocumentWindow)
{
    ui->setupUi(this);

	setAttribute(Qt::WA_DeleteOnClose);
	isUntitled = true;

	QList <QTextDocument*> docs;
	docs << ui->txtOutput->document() << ui->txtError->document() << ui->edit->document();

	for(int i=0; i<docs.size(); i++)
	{
		QTextDocument* doc = docs.at(i);

		doc->setDefaultFont( QFont("Mono") );

		QTextOption to = doc->defaultTextOption();
		to.setTabStop(to.tabStop() / 2.0);
		doc->setDefaultTextOption(to);
	}

	QList<int> ss;
	ss << height() * 0.7 << height() * 0.3;
	ui->splitter->setSizes(ss);

	connect(&thread, SIGNAL(printOutput(QString)), this, SLOT(printOutput(QString)));
	connect(&thread, SIGNAL(printError(QString)), this, SLOT(printError(QString)));

}

DocumentWindow::~DocumentWindow()
{
    delete ui;
}

void DocumentWindow::printOutput(const QString& txt)
{
	QString t = ui->txtOutput->toPlainText();
	t.append(txt);
	ui->txtOutput->setPlainText(t);
}

void DocumentWindow::printError(const QString& txt)
{
	QString t = ui->txtError->toPlainText();
	t.append(txt);
	ui->txtError->setPlainText(t);
}

QTextEdit* DocumentWindow::textEdit()
{
	return ui->edit->textEdit();
}

QTextDocument* DocumentWindow::document() const
{
	return ui->edit->document();
}

void DocumentWindow::copy()
{
	ui->edit->copy();
}
void DocumentWindow::paste()
{
	ui->edit->paste();
}
void DocumentWindow::cut()
{
	ui->edit->cut();
}




static 	int pushtraceback(lua_State* L)
{
	lua_getglobal(L, "debug");
	if (!lua_istable(L, -1)) {
		lua_pop(L, 1);
		return 1;
	}
	lua_getfield(L, -1, "traceback");
	if (!lua_isfunction(L, -1)) {
		lua_pop(L, 2);
		return 1;
	}
	lua_remove(L, 1); //remove debug table
	return 0;
}


void DocumentWindow::run()
{
	lua_State* L = lua_open();
	luaL_openlibs(L);
	pushtraceback(L);

	ui->txtOutput->clear();
	ui->txtError->clear();


	QSettings settings("Mercer", "MagLuaFrontend");
	QStringList mods = settings.value("modules", QStringList()).toStringList();
	QStringList failList;

	if(load_libs(L, mods, failList))
	{
		QErrorMessage em;
		em.showMessage("Failed to load some modules MAKE THIS MESSAGE BETTER");
		return;
	}

	ui->tabWidget->setCurrentWidget(ui->tabOutput);

	bool ok = true;

	if(isUntitled)
	{
		ok = luaL_loadstring(L, ui->edit->document()->toPlainText().toStdString().c_str());
	}
	else
	{
		if(!maybeSave(false))
		{
			lua_close(L);
			return;
		}

		ok = luaL_loadfile(L, curFile.toStdString().c_str());
	}


	thread.execute(L);

}

QTextCursor DocumentWindow::textCursor () const
{
	return ui->edit->textEdit()->textCursor();
}


void DocumentWindow::newFile()
{
	static int sequenceNumber = 1;

	isUntitled = true;
	curFile = tr("script-%1.lua").arg(sequenceNumber++);
	setWindowTitle(curFile + "[*]");

	connect(document(), SIGNAL(contentsChanged()),
			this, SLOT(documentWasModified()));
}

bool DocumentWindow::loadFile(const QString &fileName)
{
	QFile file(fileName);
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
		QMessageBox::warning(this, tr("MagLua"),
							 tr("Cannot read file %1:\n%2.")
							 .arg(fileName)
							 .arg(file.errorString()));
		return false;
	}

	QTextStream in(&file);
	QApplication::setOverrideCursor(Qt::WaitCursor);
	ui->edit->textEdit()->setPlainText(in.readAll());
	QApplication::restoreOverrideCursor();

	setCurrentFile(fileName);

	connect(document(), SIGNAL(contentsChanged()),
			this, SLOT(documentWasModified()));

	return true;
}

bool DocumentWindow::save()
{
	if (isUntitled) {
		return saveAs();
	} else {
		return saveFile(curFile);
	}
}

bool DocumentWindow::saveAs()
{
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"),
													curFile);
	if (fileName.isEmpty())
		return false;

	return saveFile(fileName);
}

bool DocumentWindow::saveFile(const QString &fileName)
{
	QFile file(fileName);
	if (!file.open(QFile::WriteOnly | QFile::Text)) {
		QMessageBox::warning(this, tr("MagLua"),
							 tr("Cannot write file %1:\n%2.")
							 .arg(fileName)
							 .arg(file.errorString()));
		return false;
	}

	QTextStream out(&file);
	QApplication::setOverrideCursor(Qt::WaitCursor);
	out << ui->edit->textEdit()->toPlainText();
	QApplication::restoreOverrideCursor();

	setCurrentFile(fileName);
	return true;
}

QString DocumentWindow::userFriendlyCurrentFile()
{
	return strippedName(curFile);
}

void DocumentWindow::closeEvent(QCloseEvent *event)
{
	if (maybeSave()) {
		event->accept();
	} else {
		event->ignore();
	}
}

void DocumentWindow::documentWasModified()
{
	setWindowModified(document()->isModified());
}

bool DocumentWindow::maybeSave(bool showDiscard)
{
	if (document()->isModified())
	{

		QMessageBox::StandardButton ret;
		if(showDiscard)
		{
			ret = QMessageBox::warning(this, tr("MagLua"),
								   tr("'%1' has been modified.\n"
									  "Do you want to save your changes?")
								   .arg(userFriendlyCurrentFile()), QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
		}
		else
		{
			ret = QMessageBox::warning(this, tr("MagLua"),
								   tr("'%1' has been modified.\n"
									  "Do you want to save your changes?")
								   .arg(userFriendlyCurrentFile()), QMessageBox::Save | QMessageBox::No);
		}
		if (ret == QMessageBox::Save)
			return save();
		else if (ret == QMessageBox::Cancel)
			return false;
	}
	return true;
}

void DocumentWindow::setCurrentFile(const QString &fileName)
{
	curFile = QFileInfo(fileName).canonicalFilePath();
	isUntitled = false;
	ui->edit->document()->setModified(false);
	setWindowModified(false);
	setWindowTitle(userFriendlyCurrentFile() + "[*]");
}

QString DocumentWindow::strippedName(const QString &fullFileName)
{
	return QFileInfo(fullFileName).fileName();
}
