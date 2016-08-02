#ifndef DOCUMENTWINDOW_H
#define DOCUMENTWINDOW_H

#include <QMainWindow>
#include <QTextCursor>
#include <QTextEdit>
#include "QLineNumberTextEdit.h"
#include <LuaThread.h>

namespace Ui {
    class DocumentWindow;
}

class DocumentWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit DocumentWindow(QWidget *parent = 0);
    ~DocumentWindow();

	void newFile();
	bool loadFile(const QString &fileName);
	bool save();
	bool saveAs();
	bool saveFile(const QString &fileName);
	QString userFriendlyCurrentFile();
	QString currentFile() { return curFile; }

	QTextCursor textCursor () const;

	QTextDocument* document() const;
	QTextEdit* textEdit();

protected:
	void closeEvent(QCloseEvent* event);
	void keyPressEvent(QKeyEvent* event);

public slots:
	void copy();
	void paste();
	void cut();
	void run();
	void stop();
	void find();
	void hidefind();

	void findNext();
	void findPrev();
	void replaceNext();
	void replacePrev();
	void replaceAll();

	void currentLineChange(int line, const QString& src);

private slots:
	void documentWasModified();
	void printOutput(const QString& txt);
	void printError(const QString& txt);

private:
	bool maybeSave(bool showDiscard=true);
	void setCurrentFile(const QString &fileName);
	QString strippedName(const QString &fullFileName);

	QString curFile;
	bool isUntitled;


	QAction* actFindNext;
	QAction* actFindPrev;
	QAction* actReplaceNext;
	QAction* actReplacePrev;

	//LuaThread thread;

private:
    Ui::DocumentWindow *ui;
};

#endif // DOCUMENTWINDOW_H
