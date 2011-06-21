#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSignalMapper>
#include "DocumentWindow.h"
#include <QMdiSubWindow>

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
	void closeEvent(QCloseEvent *event);

private slots:
	void newFile();
	void open();
	void openFile(const QString& filename);
	void save();
	void saveAs();
	void cut();
	void run();
	void copy();
	void paste();
	void about();
	void updateMenus();
	void chooseModules();
	void updateWindowMenu();
	DocumentWindow* createDocumentChild();
	void setActiveSubWindow(QWidget *window);
	void openRecentFile();

private:
	void createActions();
	void createMenus();
	void createToolBars();
	void createStatusBar();
	void readSettings();
	void writeSettings();
	void updateRecentFileActions();
	QString strippedName(const QString &fullFileName);

	DocumentWindow* activeMdiChild();
	QMdiSubWindow* findMdiChild(const QString &fileName);


	QSignalMapper *windowMapper;
    Ui::MainWindow *ui;

	QMenu *fileMenu;
	QMenu *editMenu;
	QMenu *windowMenu;
	QMenu *helpMenu;
	QToolBar *fileToolBar;
	QToolBar *editToolBar;
	QAction *modAct;
	QAction *newAct;
	QAction *openAct;
	QAction *saveAct;
	QAction *saveAsAct;
	QAction *exitAct;
	QAction *cutAct;
	QAction *copyAct;
	QAction *runAct;
	QAction *pasteAct;
	QAction *closeAct;
	QAction *closeAllAct;
	QAction *tileAct;
	QAction *cascadeAct;
	QAction *nextAct;
	QAction *previousAct;
	QAction *separatorAct;
	QAction *aboutAct;
	QAction *aboutQtAct;

	enum { MaxRecentFiles = 5 };
	QAction *recentFileActs[MaxRecentFiles];
};

#endif // MAINWINDOW_H
