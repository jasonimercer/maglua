#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "jobwidget.h"

class QAction;
class QMenu;
class QMdiArea;
class QMdiSubWindow;
class QSignalMapper;

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
//    ~MainWindow();

protected:
	void closeEvent(QCloseEvent *event);

private slots:
	void newJob();
	void open();
	void save();
	void saveAs();
//	void cut();
//	void copy();
//	void paste();
//	void about();
	void updateMenus();
	void updateWindowMenu();
	JobWidget *createMdiChild();
	void switchLayoutDirection();
	void setActiveSubWindow(QWidget *window);

private:
	void createActions();
	void createMenus();
	void createToolBars();
	void createStatusBar();
	void readSettings();
	void writeSettings();

	JobWidget *activeMdiChild();
	QMdiSubWindow* findMdiChild(const QString &fileName);


    Ui::MainWindow *ui;
	QSignalMapper *windowMapper;
};

#endif // MAINWINDOW_H
