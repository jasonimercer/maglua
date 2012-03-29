#-------------------------------------------------
#
# Project created by QtCreator 2012-03-28T21:13:21
#
#-------------------------------------------------

QT       += opengl

#QT       -= gui

CONFIG += link_pkgconfig
PKGCONFIG += lua5.1

TARGET = luagraphics_qtgl
TEMPLATE = lib
INCLUDEPATH += ../luagraphics
INCLUDEPATH += ../../common/luabaseobject
INCLUDEPATH += ../../..

DEFINES += LUAGRAPHICS_QTGL_LIBRARY

SOURCES +=  DrawOpenGL.cpp \
    QTextEditItemLua.cpp \
    QLuaHilighter.cpp \
    QGraphicsItemLua.cpp \
    QGraphicsSceneLua.cpp \
    QPushButtonItemLua.cpp \
    SignalSink.cpp

HEADERS +=  luagraphics_qtgl_global.h \
            DrawOpenGL.h \
    QTextEditItemLua.h \
    QLuaHilighter.h \
    QGraphicsItemLua.h \
    QGraphicsSceneLua.h \
    QPushButtonItemLua.h \
    SignalSink.h
