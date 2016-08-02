#-------------------------------------------------
#
# Project created by QtCreator 2011-06-17T09:54:49
#
#-------------------------------------------------

QT       += core gui opengl

TARGET = MagLua
TEMPLATE = app

CONFIG += link_pkgconfig
PKGCONFIG += lua5.1

LIBS += ../LuaGraphics/libLuaGraphics.a
INCLUDEPATH += ../LuaGraphics/

SOURCES += main.cpp\
        MainWindow.cpp \
    DocumentWindow.cpp \
    QLineNumberTextEdit.cpp \
    QLuaHilighter.cpp \
    ModuleSelect.cpp \
    LoadLibs.cpp \
    LuaThread.cpp \
    DrawOpenGL.cpp \
    ChildWindow.cpp \
    Classes.cpp \
    QLuaTabWidget.cpp \
    QLuaWidget.cpp \
    QLuaLabel.cpp \
    QLuaPushButton.cpp \
    QLuaLayout.cpp \
    QLuaSplitter.cpp \
    QLuaLineEdit.cpp \
    QLuaSettings.cpp \
    QLuaToolbar.cpp \
    QLuaAction.cpp \
    QLuaMenu.cpp \
    QLuaTimer.cpp

HEADERS  += MainWindow.h \
    DocumentWindow.h \
    QLineNumberTextEdit.h \
    QLuaHilighter.h \
    ModuleSelect.h \
    LoadLibs.h \
    LuaThread.h \
    DrawOpenGL.h \
    ChildWindow.h \
    Classes.h \
    QLuaTabWidget.h \
    QLuaWidget.h \
    QLuaLabel.h \
    QLuaPushButton.h \
    QLuaLayout.h \
    QLuaSplitter.h \
    QLuaLineEdit.h \
    QLuaSettings.h \
    QLuaToolbar.h \
    QLuaAction.h \
    QLuaMenu.h \
    QLuaTimer.h

FORMS    += MainWindow.ui \
    DocumentWindow.ui \
    ModuleSelect.ui \
    ChildWindow.ui

RESOURCES += \
    resources.qrc
