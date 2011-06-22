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
	void closeEvent(QCloseEvent *event);

public slots:
	void copy();
	void paste();
	void cut();
	void run();
	void stop();

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


	LuaThread thread;

private:
    Ui::DocumentWindow *ui;
};

#endif // DOCUMENTWINDOW_H
