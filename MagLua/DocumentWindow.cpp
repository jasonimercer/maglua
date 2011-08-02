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

	ui->frameFind->hide();

	actFindNext = new QAction(QIcon(":/icons/next.png"), tr("&Find Next"), this);
	actFindNext->setShortcut(QKeySequence::FindNext);
	actFindNext->setStatusTip("Find Next");
	connect(actFindNext, SIGNAL(triggered()), this, SLOT(findNext()));

	actFindPrev = new QAction(QIcon(":/icons/prev.png"), tr("&Find Previous"), this);
	actFindPrev->setShortcut(QKeySequence::FindPrevious);
	actFindPrev->setStatusTip("Find Previous");
	connect(actFindPrev, SIGNAL(triggered()), this, SLOT(findPrev()));

	actReplaceNext = new QAction(QIcon(":/icons/next.png"), tr("&Replace Next"), this);
	connect(actReplaceNext, SIGNAL(triggered()), this, SLOT(replaceNext()));

	actReplacePrev = new QAction(QIcon(":/icons/prev.png"), tr("&Replace Previous"), this);
	connect(actReplacePrev, SIGNAL(triggered()), this, SLOT(replacePrev()));

	connect(ui->cmdReplaceAll, SIGNAL(pressed()), this, SLOT(replaceAll()));
	connect(ui->txtFind, SIGNAL(returnPressed()), this, SLOT(findNext()));

	ui->cmdFindNext->setDefaultAction(actFindNext);
	ui->cmdFindPrev->setDefaultAction(actFindPrev);
	ui->cmdReplaceNext->setDefaultAction(actReplaceNext);
	ui->cmdReplacePrev->setDefaultAction(actReplacePrev);


	QList<int> ss;
	ss << height() * 0.7 << height() * 0.3;
	ui->splitter->setSizes(ss);

	connect(&thread, SIGNAL(printOutput(QString)), this, SLOT(printOutput(QString)));
	connect(&thread, SIGNAL(printError(QString)), this, SLOT(printError(QString)));
	connect(&thread, SIGNAL(currentLineChange(int,QString)), this, SLOT(currentLineChange(int,QString)));
	connect(ui->cmdCloseFind, SIGNAL(pressed()), this, SLOT(hidefind()));

	ui->txtError->setReadOnly(true);
	ui->txtOutput->setReadOnly(true);
}

void DocumentWindow::currentLineChange(int line, const QString& src)
{
	if(src.startsWith("@"))
	{
		QString s = src;
		s.remove(0, 1);
		if(s == curFile)
		{
			ui->edit->setBoldLine(line);
		}
	}
	else
	{
		ui->edit->setBoldLine(line);
	}
//	printf("'%s' '%s'\n", src.toStdString().c_str(), curFile.toStdString().c_str());
}

void DocumentWindow::findNext()
{
	QString txt = ui->txtFind->text();
	if(txt.length() == 0)
		return;

	QTextCursor ff = document()->find(txt, ui->edit->textEdit()->textCursor());

	if(!ff.isNull())
	{
		ui->edit->textEdit()->setTextCursor(ff);
		ui->edit->textEdit()->ensureCursorVisible();
	}
}

void DocumentWindow::findPrev()
{
	QString txt = ui->txtFind->text();
	if(txt.length() == 0)
		return;

	QTextCursor ff = document()->find(txt, ui->edit->textEdit()->textCursor(), QTextDocument::FindBackward);

	if(!ff.isNull())
	{
		ui->edit->textEdit()->setTextCursor(ff);
		ui->edit->textEdit()->ensureCursorVisible();
	}
}

void DocumentWindow::replaceNext()
{
	QString txt = ui->txtFind->text();
	if(txt.length() == 0)
		return;

	QTextCursor ff = document()->find(txt, ui->edit->textEdit()->textCursor());

	if(!ff.isNull())
	{
		ui->edit->textEdit()->setTextCursor(ff);
		ui->edit->textEdit()->insertPlainText(ui->txtReplace->text());
		ui->edit->textEdit()->ensureCursorVisible();
	}
}

void DocumentWindow::replacePrev()
{
	QString txt = ui->txtFind->text();
	if(txt.length() == 0)
		return;

	QTextCursor ff = document()->find(txt, ui->edit->textEdit()->textCursor(), QTextDocument::FindBackward);

	if(!ff.isNull())
	{
		ui->edit->textEdit()->setTextCursor(ff);
		ui->edit->textEdit()->textCursor().insertText(ui->txtReplace->text());
		ui->edit->textEdit()->ensureCursorVisible();
	}
}


void DocumentWindow::replaceAll()
{
	QString txt = ui->txtFind->text();
	if(txt.length() == 0)
		return;

	ui->edit->textEdit()->moveCursor(QTextCursor::Start);
	QTextCursor ff;

	do
	{
		ff = document()->find(txt, ui->edit->textEdit()->textCursor());

		if(!ff.isNull())
		{
			ui->edit->textEdit()->setTextCursor(ff);
			ui->edit->textEdit()->textCursor().insertText(ui->txtReplace->text());
			ui->edit->textEdit()->ensureCursorVisible();
		}
	}while(!ff.isNull());
}



DocumentWindow::~DocumentWindow()
{
    delete ui;
}

void DocumentWindow::hidefind()
{
	if(ui->frameFind->isVisible())
	{
		ui->edit->textEdit()->setFocus();
		ui->frameFind->hide();
		return;
	}
}


void DocumentWindow::keyPressEvent(QKeyEvent* event)
{
	if( event->key() == Qt::Key_Escape && ui->frameFind->isVisible())
	{
		ui->edit->textEdit()->setFocus();
		ui->frameFind->hide();
		return;
	}

	QMainWindow::keyPressEvent(event);
}


void DocumentWindow::printOutput(const QString& txt)
{
	ui->tabWidget->setCurrentWidget(ui->tabOutput);
	ui->txtOutput->moveCursor(QTextCursor::End);
	ui->txtOutput->textCursor().insertText(txt);
}

void DocumentWindow::printError(const QString& txt)
{
	ui->tabWidget->setCurrentWidget(ui->tabError);
	ui->txtError->moveCursor(QTextCursor::End);
	ui->txtError->textCursor().insertText(txt);
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

static int l_info(lua_State* L)
{
	QString msg = "MagLua by Jason Mercer (c) 2011";
	QString pre = "";
	if(lua_isstring(L, 1))
	{
		pre = lua_tostring(L, 1);
	}

	pre.append(msg);

	lua_pushstring(L, pre.toStdString().c_str());
	return 1;
}

void DocumentWindow::run()
{
	if(thread.running())
		return;

	lua_State* L = lua_open();
	luaL_openlibs(L);
	pushtraceback(L);

	ui->txtOutput->clear();
	ui->txtError->clear();

	lua_pushcfunction(L, l_info);
	lua_setglobal(L, "info");

	if(!isUntitled)
	{
		QDir::setCurrent(QFileInfo(curFile).path());
	}

	QSettings settings("Mercer", "MagLuaFrontend");
	QStringList mods = settings.value("modules", QStringList()).toStringList();
	QStringList failList;

	if(load_libs(L, mods, failList))
	{
		QErrorMessage em;
		em.showMessage("Failed to load some modules MAKE THIS MESSAGE BETTER");
		lua_close(L);
		return;
	}

	QString args = QString("arg = {%1}").arg(ui->txtArgs->text());
	if(luaL_dostring(L, args.toStdString().c_str()))
	{
		QErrorMessage em;
		em.showMessage(QString("Error in arguments: %1").arg(lua_tostring(L, -1)));
		lua_close(L);
		return;
	}

	QString aa;
	if(isUntitled)
	{
		aa = QString("argv = {}\n argv[1]=\"MagLua\"\nargv[2]=\"%2\"\n for k,v in pairs(arg) do argv[k+2]=v end").arg("untitled");
	}
	else
	{
		aa = QString("argv = {}\n argv[1]=\"MagLua\"\nargv[2]=\"%2\"\n for k,v in pairs(arg) do argv[k+2]=v end").arg(curFile);
	}

	aa.append(QString("\nargc=table.maxn(argv)"));

	if(luaL_dostring(L, aa.toStdString().c_str()))
	{
		QErrorMessage em;
		em.showMessage(QString("Error creating argv/argc: %1").arg(lua_tostring(L, -1)));
		lua_close(L);
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

void DocumentWindow::stop()
{
	thread.stop();
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


	QSettings settings("Mercer", "MagLuaFrontend");

	QStringList args = settings.value("recentArgList").toStringList();

	//	files.removeAll(fileName);
	//	files.prepend(fileName);
	//	while (files.size() > MaxRecentFiles)
	//		files.removeLast();

	QRegExp re("(.*):::(.+)");
	for(int i=0; i<args.size(); i++)
	{
		if(re.exactMatch(args.at(i)))
		{
			if(fileName == re.cap(1))
			{
				ui->txtArgs->setText(re.cap(2));
				break;
			}
		}
	}

	return true;
}

void DocumentWindow::find()
{
	ui->frameFind->show();

	QString sel = ui->edit->textEdit()->textCursor().selection().toPlainText();
	if(sel.length())
	{
		ui->txtFind->setText(sel);
	}

	ui->txtFind->setFocus();
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

	QSettings settings("Mercer", "MagLuaFrontend");

	QStringList args = settings.value("recentArgList").toStringList();

	bool foundMatch = false;
	QRegExp re("(.*):::(.*)");
	for(int i=0; i<args.size(); i++)
	{
		if(re.exactMatch(args.at(i)))
		{
			if(fileName == re.cap(1))
			{
				foundMatch = true;
				QString d = QString("%1:::%2").arg(fileName).arg(ui->txtArgs->text());
				args[i] = d;
				break;
			}
		}
	}

	if(!foundMatch)
	{
		QString d = QString("%1:::%2").arg(fileName).arg(ui->txtArgs->text());
		args.prepend(d);
	}

	while(args.size() > 20)
	{
		args.removeLast();
	}

	settings.setValue("recentArgList", args);

	//	files.removeAll(fileName);
	//	files.prepend(fileName);
	//	while (files.size() > MaxRecentFiles)
	//		files.removeLast();

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
