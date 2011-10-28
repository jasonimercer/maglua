#include "factory.h"
#include <iostream>


vector<FactoryItem*>* Factory_::items = 0;
// Factory_* staticFactory = 0;

FactoryItem::FactoryItem(int ID, newFunction Func, pushFunction Push, string Name)
{
	id = ID;
	func = Func;
	name = Name;
	push = Push;
}


void Factory_::init()
{
	if(!items)
		items = new vector<FactoryItem*>();
}


int Factory_::registerItem(int id, newFunction func, pushFunction push, string name)
{
	init();
	
// 	cout << "Registering " << name << endl;
// 	cout << id << endl;
	items->push_back(new FactoryItem(id, func, push, name));
}

void Factory_::cleanup()
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

Encodable* Factory_::newItem(int id)
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

void Factory_::lua_pushItem(lua_State* L, Encodable* item, int id)
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



