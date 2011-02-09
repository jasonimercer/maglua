#-------------------------------------------------
#
# Project created by QtCreator 2011-02-06T13:23:33
#
#-------------------------------------------------

QT       += core gui

TARGET = maglua_jobgen
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    jobwidget.cpp \
    qmagluahighlighter.cpp \
    qmagluaeditor.cpp \
	QMagLuaGraphicsView.cpp \
    QMagLuaGraphicsNode.cpp \
	QMagLuaNodeConnection.cpp \
    MagLuaNode.cpp \
    MagLuaNodeConnection.cpp \
    SimulationObject.cpp

HEADERS  += mainwindow.h \
    jobwidget.h \
    qmagluahighlighter.h \
    qmagluaeditor.h \
	QMagLuaGraphicsView.h \
    QMagLuaGraphicsNode.h \
	QMagLuaNodeConnection.h \
    MagLuaNode.h \
    MagLuaNodeConnection.h \
    SimulationObject.h

FORMS    += mainwindow.ui \
    jobwidget.ui

RESOURCES += \
    resources.qrc
