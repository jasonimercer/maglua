extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ENCODE_EXPORTS
  #define ENCODE_API __declspec(dllexport)
 #else
  #define ENCODE_API __declspec(dllimport)
 #endif
#else
 #define ENCODE_API 
#endif


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

class ENCODE_API Factory_
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
