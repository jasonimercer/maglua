#include "ChildWindow.h"
#include "ui_ChildWindow.h"
#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "QLineNumberTextEdit.h"
#include "Classes.h"
#include "QLuaLayout.h"
#include <QMdiSubWindow>

ChildWindow::ChildWindow(lua_State* _L, QWidget *parent) :
	QMainWindow(parent),
	L(_L),
	ui(new Ui::ChildWindow)
{
	refcount = 0;
	iofuncref = LUA_NOREF;
    ui->setupUi(this);

	subProxy = Singleton.mainWindow->ui->mdiArea->addSubWindow(this);

	subProxy->showNormal();
}

ChildWindow::~ChildWindow()
{
	if(iofuncref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, iofuncref);
	}

	while(children.size())
	{
		children.at(0)->refcount--;
		if(children.at(0)->refcount == 0)
			delete children.at(0);
		children.pop_front();
	}

    delete ui;
}

void ChildWindow::setIOFunction(int ref)
{
	if(iofuncref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, iofuncref);
	}
	iofuncref = ref;
}

int ChildWindow::lua_add(lua_State* L, int idx)
{
	if(!centralWidget()->layout())
		centralWidget()->setLayout(new QVBoxLayout);

	QLuaWidget* w = lua_toluawidget(L, idx);

	if(w)
	{
		centralWidget()->layout()->addWidget(w->widget);

		w->refcount++;
		children.push_back(w);

		return 0;
	}

	QLuaLayout* l = lua_tolayout(L, idx);
	if(l)
	{
		if(centralWidget()->layout())
			delete centralWidget()->layout();
		centralWidget()->setLayout(l->layout);

		l->refcount++;
		children.push_back(l);

		return 0;
	}

	return luaL_error(L, "Failed to determine type");
}


int ChildWindow::callIOFunction(int nargs)
{
	if(iofuncref != LUA_NOREF)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, iofuncref);
		lua_insert(L, -nargs-1);
		return lua_pcall(L, nargs, LUA_MULTRET, 0);
	}
	else
	{
		lua_pop(L, nargs);
		return 0;
	}
}









int lua_iswindow(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Window");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

ChildWindow* lua_towindow(lua_State* L, int idx)
{
	ChildWindow** pp = (ChildWindow**)luaL_checkudata(L, idx, "Window");
	luaL_argcheck(L, pp != NULL, idx, "`Window' expected");
	return *pp;
}

void lua_pushwindow(lua_State* L, ChildWindow* c)
{
	ChildWindow** pp = (ChildWindow**)lua_newuserdata(L, sizeof(ChildWindow**));

	*pp = c;
	luaL_getmetatable(L, "Window");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_window_new(lua_State* L)
{
	lua_pushwindow(L, new ChildWindow(L));
	return 1;
}

static int l_gc(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_iswindow(L, 1))
	{
		lua_pushstring(L, "Window");
		return 1;
	}
	return 0;
}


static int l_settitle(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->setWindowTitle(lua_tostring(L, 2));

	return 0;
}

static int l_gettitle(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	lua_pushstring(L, c->windowTitle().toStdString().c_str());

	return 1;
}

static int l_add(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;
	c->lua_add(L, 2);
	return 0;
}

static int l_setiofunc(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;
	c->setIOFunction(luaL_ref(L, LUA_REGISTRYINDEX));
	return 0;
}

static int l_setmodified(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->setWindowModified(lua_toboolean(L, 2));
	return 0;
}

static int l_show(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->show();
	c->subProxy->updateGeometry();
	return 0;
}

static int l_hide(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->hide();
	return 0;
}

static int l_maximize(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->subProxy->showMaximized();
	return 0;
}
static int l_minimize(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->subProxy->showMinimized();
	return 0;
}
static int l_normal(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	c->subProxy->showNormal();
	return 0;
}

static int l_isactivewindow(lua_State* L)
{
	ChildWindow* c = lua_towindow(L, 1);
	if(!c) return 0;

	lua_pushboolean(L, Singleton.mainWindow->ui->mdiArea->activeSubWindow() == c->subProxy);

	return 1;
}

void lua_registerwindow(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"add",        l_add},
	  {"setIOFunction", l_setiofunc},
	  {"show",      l_show},
	  {"hide",      l_hide},
	  {"title",      l_gettitle},
	  {"setTitle",   l_settitle},
	  {"setModified", l_setmodified},
	  {"maximize",    l_maximize},
	  {"minimize",    l_minimize},
	  {"normal",      l_normal},
	  {"isActiveWindow", l_isactivewindow},

	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Window");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_window_new},
		{NULL, NULL}
	};

	luaL_register(L, "Window", struct_f);
	lua_pop(L,1);
}

