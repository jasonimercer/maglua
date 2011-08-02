#-------------------------------------------------
#
# Project created by QtCreator 2011-06-17T09:54:49
#
#-------------------------------------------------

QT       += core gui

TARGET = MagLua
TEMPLATE = app

CONFIG += link_pkgconfig
PKGCONFIG += lua5.1


SOURCES += main.cpp\
        MainWindow.cpp \
    DocumentWindow.cpp \
    QLineNumberTextEdit.cpp \
    QLuaHilighter.cpp \
    ModuleSelect.cpp \
    LoadLibs.cpp \
    LuaThread.cpp

HEADERS  += MainWindow.h \
    DocumentWindow.h \
    QLineNumberTextEdit.h \
    QLuaHilighter.h \
    ModuleSelect.h \
    LoadLibs.h \
    LuaThread.h

FORMS    += MainWindow.ui \
    DocumentWindow.ui \
    ModuleSelect.ui

RESOURCES += \
    resources.qrc
