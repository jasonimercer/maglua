#-------------------------------------------------
#
# Project created by QtCreator 2012-03-28T22:28:46
#
#-------------------------------------------------

QT       += core gui opengl svg

CONFIG += link_pkgconfig
PKGCONFIG += lua5.1

TARGET = test
TEMPLATE = app

INCLUDEPATH += ../../../../
LIBS += ../../../../libMagLua.a

SOURCES += main.cpp\
        MainWindow.cpp \
    OpenGLScene.cpp \
    QFilteredGraphicsView.cpp

HEADERS  += MainWindow.h \
    OpenGLScene.h \
    QFilteredGraphicsView.h

FORMS    += MainWindow.ui
