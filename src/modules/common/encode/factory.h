extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}


#ifndef MAGLUA_FACTORY_H
#define MAGLUA_FACTORY_H

#define Factory (Factory_::Instance())
#include <vector>
#include <string>

using namespace std;

class Encodable;

class Factory_;
static Factory_* staticFactory = 0;

typedef Encodable*(*newFunction)();
typedef void (*pushFunction)(lua_State*, Encodable*);

class FactoryItem
{
public:
	FactoryItem(int ID, newFunction Func, pushFunction Push, string Name);
	int id;
	newFunction func;
	pushFunction push;
	string name;
};

class Factory_
{
public:
	static Factory_& Instance()
	{
		if(!staticFactory)
			staticFactory  = new Factory_;
// 		static Factory_ theFactory;
// 		return theFactory;
		return *staticFactory;
	}
	
	Encodable* newItem(int id);
	void lua_pushItem(lua_State* L, Encodable* item, int id);
	int registerItem(int id, newFunction func, pushFunction Push, string name);
	void cleanup();
private:
	void init();
    Factory_() {};
    Factory_(Factory_ const&) {};

	static vector<FactoryItem*>* items;
};

#endif
