#include <vector>
extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}


#ifndef sure_write
#define sure_write(fd, data, sz, ok) sure_write_(fd, data, sz, ok, __FILE__, __LINE__)
#endif
int sure_write_(int fd, void* data, int sz, bool* ok, const char* file, int line);

#ifndef sure_read
#define sure_read(fd, data, sz, ok) sure_read_(fd, data, sz, ok, __FILE__, __LINE__)
#endif
int sure_read_(int fd, void* data, int sz, bool* ok, const char* file, int line);



class LuaVariableGroup
{
public:
	LuaVariableGroup();
	~LuaVariableGroup();
	
	std::vector<int> sizes;
	std::vector<char*> variables;
	
	void add(char* data, int size);
	void clear();
	void write(int fd, bool& ok);
	void read(int fd, bool& ok);
	
	void readState(lua_State* L);
	void writeState(lua_State* L);
};
