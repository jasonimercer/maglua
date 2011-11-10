#include <vector>
#include <string>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}


// given the table "module_path" in L, make a list of all the 
// libraries to load.
void getModulePaths(lua_State* L, std::vector<std::string>& module_paths);
void getModuleDirectories(std::vector<std::string>& mds, std::vector<std::string>& initial_args);
