#-------------------------------------------------
#
# Project created by QtCreator 2012-03-28T21:13:21
#
#-------------------------------------------------

QT       += core gui opengl svg

unix {
CONFIG += link_pkgconfig
PKGCONFIG += lua5.1
}
win32 {
INCLUDEPATH += ../../../../Common
LIBS += -L../../../../Common
LIBS += -lluabaseobject
LIBS += ../../../../Common/luagraphics.lib
LIBS += -lluagraphics
LIBS += -llua-5.1_x86_64_compat
}

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
    SignalSink.cpp \
    QItemLua.cpp \
    QSliderItemLua.cpp

HEADERS +=  luagraphics_qtgl_global.h \
            DrawOpenGL.h \
    QTextEditItemLua.h \
    QLuaHilighter.h \
    QGraphicsItemLua.h \
    QGraphicsSceneLua.h \
    QPushButtonItemLua.h \
    SignalSink.h \
    QItemLua.h \
    QSliderItemLua.h


