#-------------------------------------------------
#
# Project created by QtCreator 2012-03-28T22:28:46
#
#-------------------------------------------------

QT       += core gui opengl svg


win32 {
INCLUDEPATH += ../../../../../Common
LIBS += -L../../../../../Common
LIBS += -lluabaseobject
LIBS += ../../../../../Common/luagraphics.lib
LIBS += -lluagraphics
LIBS += -llua-5.1_x86_64_compat
LIBS += -llibfftw3-3
LIBS += ../../../../../Common/libMagLua.lib
}
unix {
CONFIG += link_pkgconfig
PKGCONFIG += lua5.1
LIBS += ../../../../libMagLua.a
}


TARGET = test
TEMPLATE = app

INCLUDEPATH += ../../../../

SOURCES += main.cpp\
        MainWindow.cpp \
    OpenGLScene.cpp \
    QFilteredGraphicsView.cpp

HEADERS  += MainWindow.h \
    OpenGLScene.h \
    QFilteredGraphicsView.h

FORMS    += MainWindow.ui
