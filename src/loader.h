#ifndef LOADITEM_H
#define LOADITEM_H

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <string>
#include <vector>

class loader_item
{
public:
	loader_item();
	loader_item(const std::string fullpath);
	loader_item(const loader_item& ls);

	std::string filename;
	std::string fullpath;
	std::string truename;
	std::string error;
	bool registered;
	int main_return;
	int version;
};

int load_items(lua_State* L, std::vector<loader_item>& items, int argc, char** argv, int quiet);

#endif
