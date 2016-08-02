#include "QLuaAction.h"
#include "MainWindow.h"
#include <iostream>
#include <QKeySequence>
using namespace std;

QLuaAction::QLuaAction(lua_State* _L, QAction* act) :
	QObject(act)
{
	L = _L;
	refcount = 0;
	action = act;
	triggerref = LUA_NOREF;

	connect(action, SIGNAL(triggered(bool)), this, SLOT(triggered(bool)));
}

QLuaAction::~QLuaAction()
{
	if(triggerref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, triggerref);
	}
}

void QLuaAction::triggered(bool b)
{
	if(triggerref != LUA_NOREF)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, triggerref);
		lua_pushboolean(L, b);

		if(lua_pcall(L, 1, 0, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
			QErrorMessage* msg = new QErrorMessage(Singleton.mainWindow);
			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
			lua_pop(L, lua_gettop(L));
		}
		lua_gc(L, LUA_GCCOLLECT, 0);
	}
}

void QLuaAction::triggered()
{

}


void QLuaAction::setTriggerFunction(int ref)
{
	if(triggerref != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, triggerref);
	}
	triggerref = ref;
}





int lua_isaction(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Action");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaAction* lua_toluaaction(lua_State* L, int idx)
{
	QLuaAction** pp = (QLuaAction**)luaL_checkudata(L, idx, "Action");
	luaL_argcheck(L, pp != NULL, idx, "`Action' expected");
	return *pp;
}

QAction* lua_toaction(lua_State* L, int idx)
{
	QLuaAction* pp = lua_toluaaction(L, idx);
	if(pp)
		return (QAction*)(pp->action);
	return 0;
}

void lua_pushluaaction(lua_State* L, QLuaAction* c)
{
	QLuaAction** pp = (QLuaAction**)lua_newuserdata(L, sizeof(QLuaAction**));

	*pp = c;
	luaL_getmetatable(L, "Action");
	lua_setmetatable(L, -2);
	c->refcount++;
}

void lua_pushaction(lua_State* L, QAction* c)
{
	lua_pushluaaction(L, new QLuaAction(L, c));
}

static int l_action_new(lua_State* L)
{
	QString txt = lua_tostring(L, 1);
	lua_pushluaaction(L, new QLuaAction(L, new QAction(txt, Singleton.mainWindow)));
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_isaction(L, 1))
	{
		lua_pushstring(L, "Action");
		return 1;
	}
	return 0;
}



static int l_settrigger(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;

	lua_pushvalue(L, 2);
	c->setTriggerFunction(luaL_ref(L, LUA_REGISTRYINDEX));
	return 0;
}
static int l_setchecked(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	c->action->setChecked(lua_toboolean(L, 2));
	return 0;
}
static int l_setenabled(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	c->action->setEnabled(lua_toboolean(L, 2));
	return 0;
}
static int l_setvisible(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	c->action->setVisible(lua_toboolean(L, 2));
	return 0;
}
static int l_settext(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	c->action->setText(lua_tostring(L, 2));
	return 0;
}
static int l_seticon(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	c->action->setIcon(QIcon(lua_tostring(L, 2)));
	return 0;
}
static int l_setshortcut(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;

	if(lua_isnumber(L, 2))
	{
		c->action->setShortcut(QKeySequence( (QKeySequence::StandardKey)lua_tointeger(L, 2)));
	}
	else
	{
		c->action->setShortcut(QKeySequence(lua_tostring(L, 2)));
	}
	return 0;
}




static int l_getchecked(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	lua_pushboolean(L, c->action->isChecked());
	return 1;
}
static int l_getenabled(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	lua_pushboolean(L, c->action->isEnabled());
	return 1;
}
static int l_getvisible(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	lua_pushboolean(L, c->action->isVisible());
	return 1;
}
static int l_gettext(lua_State* L)
{
	QLuaAction* c = lua_toluaaction(L, 1);
	if(!c) return 0;
	lua_pushstring(L, c->action->text().toStdString().c_str());
	return 1;
}

void lua_registeraction(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"setFunction",  l_settrigger},

	  {"setChecked",   l_setchecked},
	  {"setEnabled",   l_setenabled},
	  {"setVisible",   l_setvisible},
	  {"setText",      l_settext},
	  {"setIcon",      l_seticon},
	  {"setShortCut",  l_setshortcut},

	  {"isChecked",   l_getchecked},
	  {"isEnabled",   l_getenabled},
	  {"isVisible",   l_getvisible},
	  {"text",        l_gettext},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "Action");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_action_new},
		{NULL, NULL}
	};

	luaL_register(L, "Action", struct_f);
	lua_pop(L,1);

#define _F(xyz) lua_pushstring(L, #xyz); lua_pushinteger(L, (int)QKeySequence::xyz); lua_settable(L, -3);

	lua_newtable(L);
	_F(AddTab);               	_F(Back);                 	_F(Bold);
	_F(Close);                	_F(Copy);                 	_F(Cut);
	_F(Delete);               	_F(DeleteEndOfLine);      	_F(DeleteEndOfWord);
	_F(DeleteStartOfWord);    	_F(Find);                 	_F(FindNext);
	_F(FindPrevious);         	_F(Forward);              	_F(HelpContents);
	_F(InsertLineSeparator);  	_F(InsertParagraphSeparator);	_F(Italic);
	_F(MoveToEndOfBlock);     	_F(MoveToEndOfDocument);  	_F(MoveToEndOfLine);
	_F(MoveToNextChar);       	_F(MoveToNextLine);       	_F(MoveToNextPage);
	_F(MoveToNextWord);       	_F(MoveToPreviousChar);   	_F(MoveToPreviousLine);
	_F(MoveToPreviousPage);   	_F(MoveToPreviousWord);   	_F(MoveToStartOfBlock);
	_F(MoveToStartOfDocument);	_F(MoveToStartOfLine);    	_F(New);
	_F(NextChild);            	_F(Open);                 	_F(Paste);
	_F(Preferences);          	_F(PreviousChild);        	_F(Print);
	_F(Quit);                 	_F(Redo);                 	_F(Refresh);
	_F(Replace);              	_F(SaveAs);               	_F(Save);
	_F(SelectAll);            	_F(SelectEndOfBlock);     	_F(SelectEndOfDocument);
	_F(SelectEndOfLine);      	_F(SelectNextChar);       	_F(SelectNextLine);
	_F(SelectNextPage);       	_F(SelectNextWord);       	_F(SelectPreviousChar);
	_F(SelectPreviousLine);   	_F(SelectPreviousPage);   	_F(SelectPreviousWord);
	_F(SelectStartOfBlock);   	_F(SelectStartOfDocument);	_F(SelectStartOfLine);
	_F(Underline);            	_F(Undo);                 	_F(UnknownKey);
	_F(WhatsThis);            	_F(ZoomIn);               	_F(ZoomOut);
	lua_setglobal(L, "QKeySequence");
}
