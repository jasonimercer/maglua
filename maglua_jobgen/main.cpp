#include <QtGui/QApplication>
#include "mainwindow.h"

#include "QMagLuaGraphicsNode.h"

int main(int argc, char *argv[])
{
	QMagLuaGraphicsNode n;
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
