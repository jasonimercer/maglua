#include "factory.h"
#include <iostream>
#include <vector>
#include <string>
#include <string.h>

using namespace std;



class FactoryItem
{
public:
	FactoryItem(int ID, newFactoryFunction Func, pushFunction Push, const char* Name);
	int id;
	newFactoryFunction func;
	pushFunction push;
	char* name;
};

class Factory
{
public:
	Factory() {}

	LuaBaseObject* newItem(int id);
	const char* getName(int id);
	void lua_pushItem(lua_State* L, LuaBaseObject* item, int id);
	int registerItem(int id, newFactoryFunction func, pushFunction Push, const char* name);
	void cleanup();
private:
	void init();
	vector<FactoryItem*>* items;
};

static Factory theFactory;

FactoryItem::FactoryItem(int ID, newFactoryFunction Func, pushFunction Push, const char* Name)
{
	id = ID;
	func = Func;
	push = Push;

        const int n = strlen(Name) + 1;
        name = new char[n];
        memcpy(name, Name, n);
}

void Factory::init()
{
	if(!items)
		items = new vector<FactoryItem*>();
}


int Factory::registerItem(int id, newFactoryFunction func, pushFunction push, const char* name)
{
	init();
	
// 	cout << "Registering " << name << endl;
// 	cout << id << endl;
	items->push_back(new FactoryItem(id, func, push, name));
	return 0;
}

void Factory::cleanup()
{
	if(items)
	{
		while(items->size())
		{
			delete items->back();
			items->pop_back();
		}
	}
	items = 0;
}

LuaBaseObject* Factory::newItem(int id)
{
	init();
	for(unsigned int i=0; i<items->size(); i++)
	{
		if(items->at(i)->id == id)
		{
			return items->at(i)->func();
		}
	}
	return 0;
}

const char* Factory::getName(int id)
{
	init();
	for(unsigned int i=0; i<items->size(); i++)
	{
		if(items->at(i)->id == id)
		{
		    return items->at(i)->name;
		}
	}
	return "Unknown";
}

void Factory::lua_pushItem(lua_State* L, LuaBaseObject* item, int id)
{
	init();
	for(unsigned int i=0; i<items->size(); i++)
	{
		if(items->at(i)->id == id)
		{
			items->at(i)->push(L, item);
			return;
		}
	}

	
	fprintf(stderr, "Failed to find pushFunction for id = %i\n", id);
}

#include "MurmurHash3.h"

int hash32(const char* string)
{
    int N = strlen(string);
    int seed = 1000;
    int res;

    MurmurHash3_x86_32(string, N, seed, &res);
    return res;
}




const char* Factory_typename(int id)
{
	return theFactory.getName(id);
}


LuaBaseObject* Factory_newItem(int id)
{
	return theFactory.newItem(id);
}

void Factory_lua_pushItem(lua_State* L, LuaBaseObject* item, int id)
{
	theFactory.lua_pushItem(L, item, id);
}

int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, const char* name)
{
	return theFactory.registerItem(id, func, Push, name);
}

void Factory_cleanup()
{
	theFactory.cleanup();
}
