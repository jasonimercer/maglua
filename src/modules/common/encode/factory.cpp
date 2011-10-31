#include "factory.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;


class Encodable;

class FactoryItem
{
public:
	FactoryItem(int ID, newFactoryFunction Func, pushFunction Push, string Name);
	int id;
	newFactoryFunction func;
	pushFunction push;
	string name;
};

class Factory
{
public:
    Factory() {};

	Encodable* newItem(int id);
	void lua_pushItem(lua_State* L, Encodable* item, int id);
	int registerItem(int id, newFactoryFunction func, pushFunction Push, string name);
	void cleanup();
private:
	void init();
	vector<FactoryItem*>* items;
};

static Factory theFactory;

FactoryItem::FactoryItem(int ID, newFactoryFunction Func, pushFunction Push, string Name)
{
	id = ID;
	func = Func;
	name = Name;
	push = Push;
}

void Factory::init()
{
	if(!items)
		items = new vector<FactoryItem*>();
}


int Factory::registerItem(int id, newFactoryFunction func, pushFunction push, string name)
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

Encodable* Factory::newItem(int id)
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

void Factory::lua_pushItem(lua_State* L, Encodable* item, int id)
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



ENCODE_API Encodable* Factory_newItem(int id)
{
	return theFactory.newItem(id);
}

ENCODE_API void Factory_lua_pushItem(lua_State* L, Encodable* item, int id)
{
	theFactory.lua_pushItem(L, item, id);
}

ENCODE_API int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, std::string name)
{
	return theFactory.registerItem(id, func, Push, name);
}

ENCODE_API void Factory_cleanup()
{
	theFactory.cleanup();
}
