#include "LuaThread.h"
#include <QTextEdit>
#include <stdio.h>
#include <QDir>
#include <QFileInfo>
#include <QTimer>
#include <QSettings>
#include "QLineNumberTextEdit.h"
#include "LoadLibs.h"
#include <iostream>
#include "MainWindow.h"
using namespace std;

LuaThread::LuaThread(lua_State* parentl, QObject *parent) :
    QThread(parent)
{
	parentL = parentl;
	L = 0;

	requestStop = false;

	outputFuncRef = LUA_NOREF;
	errorFuncRef = LUA_NOREF;
}

LuaThread::~LuaThread()
{
	mutex.lock();
//	abort = true;
	condition.wakeOne();
	mutex.unlock();

	if(outputFuncRef != LUA_NOREF)
	{
		luaL_unref(parentL, LUA_REGISTRYINDEX, outputFuncRef);
	}
	if(errorFuncRef != LUA_NOREF)
	{
		luaL_unref(parentL, LUA_REGISTRYINDEX, errorFuncRef);
	}

	if(L)
	{
		lua_close(L);
		L = 0;
	}
}

static void getStrings(lua_State* L, int idx, QStringList& list)
{
	if(lua_istable(L, idx))
	{
		lua_pushnil(L);
		while(lua_next(L, idx))
		{
			getStrings(L, lua_gettop(L), list);
			lua_pop(L, 1);
		}
	}
	else
	{
		if(lua_isstring(L, idx))
		{
			list.push_back(lua_tostring(L, idx));
		}
	}
}

int LuaThread::loadModules(lua_State* L)
{
	for(int i=1; i<=lua_gettop(L); i++)
	{
		getStrings(L, i, loadList);
	}

//	if(load_libs(L, list, faillist))
//	{
//		callErr("Failed to load some modules MAKE THIS MESSAGE BETTER");
//		return 1;
//	}
	return 0;
}

void LuaThread::setOutFunction(int ref)
{
	if(outputFuncRef != LUA_NOREF)
	{
		luaL_unref(parentL, LUA_REGISTRYINDEX, outputFuncRef);
	}
	outputFuncRef = ref;
}

void LuaThread::setErrFunction(int ref)
{
	if(errorFuncRef != LUA_NOREF)
	{
		luaL_unref(parentL, LUA_REGISTRYINDEX, errorFuncRef);
	}
	errorFuncRef = ref;
}

void LuaThread::callOut(const QString& msg)
{
	if(outputFuncRef != LUA_NOREF)
	{
		emit(threadedCallWithMsg(parentL, outputFuncRef, msg));
//		lua_rawgeti(parentL, LUA_REGISTRYINDEX, outputFuncRef);
//		lua_pushstring(parentL, msg.toStdString().c_str());

//		if(lua_pcall(parentL, 1, 0, 0))
//		{
//			cerr << lua_tostring(parentL, 1) << endl;
//			lua_pop(parentL, 1);
//		}
	}
}

void LuaThread::callErr(const QString& msg)
{
	//printf("%s\n", msg.toStdString().c_str());

	if(errorFuncRef != LUA_NOREF)
	{
		emit(threadedCallWithMsg(parentL, errorFuncRef, msg));

//		lua_rawgeti(parentL, LUA_REGISTRYINDEX, errorFuncRef);
//		lua_pushstring(parentL, msg.toStdString().c_str());

//		if(lua_pcall(parentL, 1, 0, 0))
//		{
//			cerr << lua_tostring(parentL, 1) << endl;
//			lua_pop(parentL, 1);
//		}
	}
}


bool LuaThread::stopRequested() const
{
	return requestStop;
}




static int print(lua_State* L)
{
	LuaThread* thread = (LuaThread*)lua_touserdata(L, lua_upvalueindex(1));
	if(!thread)
		return luaL_error(L, "LuaThread upvalue is not set");

	QString line = "";

	int n = lua_gettop(L);  /* number of arguments */
	int i;
	lua_getglobal(L, "tostring");
	for(i=1; i<=n; i++)
	{
		const char *s;
		lua_pushvalue(L, -1);
		lua_pushvalue(L, i);
		lua_call(L, 1, 1);
		s = lua_tostring(L, -1);  /* get result */
		if (s == NULL)
			return luaL_error(L, LUA_QL("tostring") " must return a string to "
							  LUA_QL("print"));
		if (i>1)
			line.append("\t");
		line.append(s);
		lua_pop(L, 1);  /* pop result */
	}
	line.append("\n");

	thread->callOut(line);

	return 0;
}

void LuaThread::setCurrentLine(int line, const QString& src)
{
	if(currentLine != line || currentSource != src)
	{
		QString currentSource;
		emit(currentLineChange(line, src));

		currentLine = line;
		currentSource = src;
	}
}


static void l_handler(lua_State *L, lua_Debug* /* ar */)
{
	lua_rawgeti(L, LUA_REGISTRYINDEX, 100);
	LuaThread* thread = (LuaThread*) lua_touserdata(L, -1);
	lua_pop(L, 1);
	if(thread)
	{
		/*
		lua_getinfo(L, "Sl", ar);
		if(ar->currentline > 0)
		{
			thread->setCurrentLine(ar->currentline, ar->source);
		}
		*/

		if(thread->stopRequested())
		{
			luaL_error(L, "Stopped by user request");
		}
	}
}

void LuaThread::run()
{
	currentLine = -1;
	if(!L)
		cerr << "L is null" << endl;
	else
	{
		if(lua_pcall(L, 0, 0, -2))
		{
			callErr(lua_tostring(L, -1));
		}
	}

//	mutex.lock();
//	lua_close(L);
//	L = 0;
//	mutex.unlock();
}

void LuaThread::stop()
{
	requestStop = true;
}

static 	int pushtraceback(lua_State* L)
{
	lua_getglobal(L, "debug");
	if (!lua_istable(L, -1)) {
		lua_pop(L, 1);
		return 1;
	}
	lua_getfield(L, -1, "traceback");
	if (!lua_isfunction(L, -1)) {
		lua_pop(L, 2);
		return 1;
	}
	lua_remove(L, 1); //remove debug table
	return 0;
}

/*
for k,v in pairs(_G) do
	print(k,v)
end
*/

static int l_info(lua_State* L)
{
	QString msg = AUTHOR_COPYRIGHT_BANNER;
	QString pre = "";
	if(lua_isstring(L, 1))
	{
		pre = lua_tostring(L, 1);
	}

	pre.append(msg);

	lua_pushstring(L, pre.toStdString().c_str());
	return 1;
}

int LuaThread::preSetup(const QString args, const QString title)
{
	if(isRunning())
		return 1;

	if(L)
	{
		lua_close(L);
		L = 0;
	}

	L = lua_open();
	luaL_openlibs(L);

	lua_pushcfunction(L, l_info);
	lua_setglobal(L, "info");
	QStringList faillist;

	if(load_libs(L, loadList, faillist))
	{
		callErr("Failed to load some modules MAKE THIS MESSAGE BETTER");
		return 1;
	}

//	QSettings settings("Mercer", "MagLuaFrontend");
//	QStringList mods = settings.value("modules", QStringList()).toStringList();
//	QStringList failList;

//	if(load_libs(L, mods, failList))
//	{
//		callErr("Failed to load some modules MAKE THIS MESSAGE BETTER");
//		lua_close(L);
//		L = 0;
//		return 1;
//	}

	QString aa = QString("arg = {%1}").arg(args);
	if(luaL_dostring(L, aa.toStdString().c_str()))
	{
		callErr(QString("Error in arguments: %1").arg(lua_tostring(L, -1)));
		lua_close(L);
		L = 0;
		return 1;
	}


	aa = QString("argv = {}\n argv[1]=\"MagLua\"\nargv[2]=\"%2\"\n for k,v in pairs(arg) do argv[k+2]=v end").arg(title);

	aa.append(QString("\nargc=table.maxn(argv)"));

	if(luaL_dostring(L, aa.toStdString().c_str()))
	{
		callErr(QString("Error creating argv/argc: %1").arg(lua_tostring(L, -1)));
		lua_close(L);
		L = 0;
		return 1;
	}

	return 0;
}


void LuaThread::doFile(const QString file, const QString args)
{
	QDir::setCurrent(QFileInfo(file).path());

	if(preSetup(args, file))
		return;

	pushtraceback(L);
	if(luaL_loadfile(L, file.toStdString().c_str()))
	{
		callErr(lua_tostring(L, -1));
		lua_close(L);
		L = 0;
		return;
	}
	execute();
}

void LuaThread::doCode(const QString code, const QString args)
{
	if(preSetup(args, "untitled"))
		return;

	pushtraceback(L);

	if(luaL_loadstring(L, code.toStdString().c_str()))
	{
		callErr(lua_tostring(L, -1));
		lua_close(L);
		L = 0;
		return;
	}
	execute();
}




void LuaThread::execute()
{
	if(!isRunning())
	{
		lua_pushlightuserdata(L, this);
		lua_pushcclosure(L, print, 1);
		lua_setglobal(L, "print");

		lua_pushlightuserdata(L, this);
		lua_rawseti(L, LUA_REGISTRYINDEX, 100);

		lua_sethook(L, l_handler, LUA_MASKCOUNT, 10);

	//	QMutexLocker locker(&mutex);

		requestStop = false;

		start(LowPriority);
	}
	else
	{
//		restart = true;
//		condition.wakeOne();
	}

//	lua_close(L);
}







int lua_isluathread(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Thread");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

LuaThread* lua_toluathread(lua_State* L, int idx)
{
	LuaThread** pp = (LuaThread**)luaL_checkudata(L, idx, "Thread");
	luaL_argcheck(L, pp != NULL, idx, "`Thread' expected");
	return *pp;
}

void lua_pushluathread(lua_State* L, LuaThread* c)
{
	LuaThread** pp = (LuaThread**)lua_newuserdata(L, sizeof(LuaThread**));
	c->connect(c, SIGNAL(threadedCallWithMsg(lua_State*,int,QString)), Singleton.mainWindow, SLOT(threadedCallWithMsg(lua_State*,int,QString)), Qt::QueuedConnection);

	*pp = c;
	luaL_getmetatable(L, "Thread");
	lua_setmetatable(L, -2);
	c->refcount++;
}

static int l_new(lua_State* L)
{
	lua_pushluathread(L, new LuaThread(L));
	return 1;
}

static int l_gc(lua_State* L)
{
	LuaThread* c = lua_toluathread(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_isluathread(L, 1))
	{
		lua_pushstring(L, "Thread");
		return 1;
	}
	return 0;
}

static int l_docode(lua_State* L)
{
	LuaThread* c = lua_toluathread(L, 1);
	if(!c) return 0;

	//code args
	c->doCode(lua_tostring(L, 2), lua_tostring(L, 3));
	return 0;
}
static int l_dofile(lua_State* L)
{
	LuaThread* c = lua_toluathread(L, 1);
	if(!c) return 0;

	//file args
	c->doFile(lua_tostring(L, 2), lua_tostring(L, 3));
	return 0;
}



static int l_setout(lua_State* L)
{
	LuaThread* c = lua_toluathread(L, 1);
	if(!c) return 0;

	c->setOutFunction(luaL_ref(L, LUA_REGISTRYINDEX));

	return 0;
}
static int l_seterr(lua_State* L)
{
	LuaThread* c = lua_toluathread(L, 1);
	if(!c) return 0;

	c->setErrFunction(luaL_ref(L, LUA_REGISTRYINDEX));

	return 0;
}
static int l_stop(lua_State* L)
{
	LuaThread* c = lua_toluathread(L, 1);
	if(!c) return 0;
	c->stop();
	return 0;
}

static int l_loadmodules(lua_State* L)
{
	LuaThread* c = lua_toluathread(L, 1);
	if(!c) return 0;

	lua_remove(L, 1);

	c->loadModules(L);

	return 0;
}

void lua_registerluathread(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"doCode", l_docode},
	  {"doFile", l_dofile},

	  {"setOutputFunction", l_setout},
	  {"setErrorFunction", l_seterr},

	  {"addModules",  l_loadmodules},

	  {"stop", l_stop},

	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Thread");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_new},
		{NULL, NULL}
	};

	luaL_register(L, "Thread", struct_f);
	lua_pop(L,1);
}
