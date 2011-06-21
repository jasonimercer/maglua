#include "LoadLibs.h"

#include <dlfcn.h>

static int load_lib(lua_State* L, QStringList& loaded, const QString& name)
{
	if(loaded.contains(name)) //then already loaded
	{
		return 3;
	}

	void* handle = dlopen(name.toStdString().c_str(),  RTLD_NOW | RTLD_GLOBAL);

	if(!handle)
	{
		// don't report error, we may be able to deal with it
		return 2;
	}


	typedef int (*lua_func)(lua_State*);


	lua_func lib_register = (lua_func) dlsym(handle, "lib_register");

	if(!lib_register)
	{
		// we'll report this later
		dlclose(handle);
		return 1;
	}

	lib_register(L); // call the register function from the library

	loaded << name;

	return 0;
}



int load_libs(lua_State* L, const QStringList& libs, QStringList& failList)
{
	failList.clear();
	QStringList loaded;

	int num_loaded_this_round;
	do
	{
		num_loaded_this_round = 0;

		for(int i=0; i<libs.size(); i++)
		{
			if(!load_lib(L, loaded, libs.at(i)))
			{
				num_loaded_this_round++;
			}
		}
	}while(num_loaded_this_round > 0);


	for(int i=0; i<libs.size(); i++)
	{
		if(!loaded.contains(libs.at(i)))
		{
			failList << libs.at(i);
		}
	}

	if(failList.size())
		return 1;

	return 0;
}
