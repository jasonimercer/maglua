/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

// This is the main entry point for maglua. It loads in the 
// modules in the default directory

#include "info.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <errno.h>
#ifdef _MPI
 #include <mpi.h>
#endif
#include "main.h"
#include "loader.h"
#include "import.h"
#include "modules.h"

using namespace std;

vector<loader_item> theModules;
vector<string> initial_args;
vector<string> moduleDirectories;

int registerLibs(int suppress, lua_State* L);
void lua_addargs(lua_State* L, int argc, char** argv); //add initial_args to state (after module consumption)
static int l_info(lua_State* L);
void print_help();
int suppress;
const char* reference();

// 
// command line switches:
//  -q                  run quietly, omit some startup info printing
//  --module_path       print primary mod dir
//  --setup mod_dir     setup startup files in $(HOME)/.maglua.d
//                        with mod_dir in the list of module paths  
//  -h --help           show this help
// 

int main(int argc, char** argv)
{
	suppress = 0; //chatter
	int shutdown = 0;
#ifdef _MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &suppress); //only rank 0 will chatter
#endif
	//record initial arguments (after MPI has had it's way with them)
	for(int i=0; i<argc; i++)
		initial_args.push_back(argv[i]);

	if(!suppress) //prevent slaves from getting legal. 
		fprintf(stderr, "This evaluation version of MagLua is for private, non-commercial use only\n");

	vector<string>::iterator it;
	for(it=initial_args.begin(); it!= initial_args.end(); ++it)
	{
		// module path can change the length of initial_args, invalidating iterators
		if((*it).compare("--module_path") == 0)
		{
			vector<string> mp;
			getModuleDirectories(mp,initial_args);
			for(unsigned int i=0; i<mp.size() && i < 1; i++)
			{
				cout << mp[i] << endl;
			}
			shutdown = 1;
			break;
		}
	}
	
	for(it=initial_args.begin(); it!= initial_args.end(); ++it)
	{
		if((*it).compare("-q") == 0)
		{
			(*it) = ""; //consume
			suppress = 1;
		}
		
		if((*it).compare("-h") == 0 || (*it).compare("--help") == 0)
		{
			print_help();
			shutdown = 1;
		}
		
		if((*it).compare("-v") == 0 || (*it).compare("--version") == 0)
		{
			printf("MagLua-r%i\n", __revi);
			shutdown = 1;
		}

		if((*it).compare("--setup") == 0)
		{
			if((it+1) != initial_args.end())
			{
				lua_State *L = lua_open();
				luaL_openlibs(L);

				if(luaL_dostring(L, setup_lua_code))
				{
					fprintf(stderr, "%s\n", lua_tostring(L, -1));
				}
				
				lua_getglobal(L, "setup");
				lua_pushstring(L, (*(it+1)).c_str());
				
				if(lua_pcall(L, 1,0,0))
				{
					fprintf(stderr, "%s\n", lua_tostring(L, -1));
				}
				
				lua_close(L);
				shutdown = 1;
			}
		}
	}
	
	if(shutdown)
	{
		#ifdef _MPI
			MPI_Finalize();
		#endif
		return 0;
	}
	
#ifdef _MPI
	{
		//suppress startup messages for non-root rank
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		suppress |= rank;
	}
#endif

	
	lua_State *L = lua_open();
	luaL_openlibs(L);

	if(!suppress)
	{
		cout << "MagLua r" << __revi << " by Jason Mercer (c) 2011" << endl;
		cout << endl;
		cout << reference() << endl;
		cout << endl;
	}


	if(!registerMain(L))
	{
		if(!suppress)
		{
			const char* printModules = 
				"print(\"Modules:\") \n"\
				"local m = {}\n"\
				"for k,v in pairs(getModules()) do\n"\
				"	table.insert(m, v.name .. \"(r\" .. v.version .. \")\")\n"\
				"end\n"\
				"table.sort(m)\n"\
				"print(table.concat(m, \", \"))";

			if(luaL_dostring(L, printModules))
			{
				fprintf(stderr, "%s\n", lua_tostring(L, -1));
				return -1;
			}
	    }	
		int script = 0;

		// execute each lua script on the command line
		// TODO: convert the following to Lua code
		for(int i=1; i<argc; i++)
		{
			const char* fn = argv[i];
			int len = strlen(fn);
			
			if(len > 4)
			{
				if(strncasecmp(fn+len-4, ".lua", 4) == 0)
				{
					script++;
					if(luaL_dofile(L, fn))
					{
						const char* errmsg = lua_tostring(L, -1);
						fprintf(stderr, "Error:\n%s\n", errmsg);
					}
				}
			}
		}
		
		if(!script)
		{
			cerr << "Please supply a MagLua script (*.lua)" << endl;
		}
		lua_close(L);
	
#ifdef _MPI
		MPI_Finalize();
#endif
	}
	

	return 0;
}

// return info about the loaded modules
static int l_modules(lua_State* L)
{
	lua_newtable(L);
	
	for(unsigned int i=0; i<theModules.size(); i++)
	{
		lua_pushinteger(L, i+1);
		lua_newtable(L);
		
		lua_pushstring(L, "filename");
		lua_pushstring(L, theModules[i].filename.c_str());
		lua_settable(L, -3);
		
		lua_pushstring(L, "fullpath");
		lua_pushstring(L, theModules[i].fullpath.c_str());
		lua_settable(L, -3);
		
		lua_pushstring(L, "name");
		lua_pushstring(L, theModules[i].truename.c_str());
		lua_settable(L, -3);
		
		lua_pushstring(L, "version");
		lua_pushinteger(L, theModules[i].version);
		lua_settable(L, -3);

		lua_settable(L, -3);
	}
	
	return 1;
}

MAGLUA_API int registerMain(lua_State* L)
{
	// add module path
	int modbase = 1;
	
	lua_newtable(L);
	{
		vector<string> m;
		getModuleDirectories(m, initial_args);
		for(unsigned int i=0; i<m.size(); i++)
		{
			lua_pushinteger(L, modbase);
			lua_pushstring(L, m[i].c_str());
			lua_settable(L, -3);
			modbase++;

#ifdef WIN32
			// need to locally add mod dirs to PATH so that it will 
			// automatically satisfy dll reqs. Hate Windows
			{
				char* p = getenv("PATH");
				string sp = p;

				if( sp.find(m[i]) == string::npos ) //then PATH doesn't have path yet
				{
					sp.append(";");
					sp.append(m[i]);
					string patheq = "PATH=";
					patheq.append(sp);
					putenv(patheq.c_str());
				}
			}
#endif
		}
	}
	lua_setglobal(L, "module_path");
	
	// add info and arguments
	lua_pushcfunction(L, l_info);
	lua_setglobal(L, "info");

	lua_pushcfunction(L, l_modules);
	lua_setglobal(L, "getModules");
	
	// get the path to each shared library
	vector<string> module_paths;
	getModulePaths(L, module_paths);

	
	// build/initialize "theModules", this will hold all info about the libraries
	theModules.clear();
	
	for(vector<string>::iterator it=module_paths.begin(); it != module_paths.end(); ++it)
	{
		theModules.push_back(loader_item( *it ));
	}
	
	// build argc, argv to look like main args
	// should probably build from argc/argv in L
	int argc = 0;//initial_args.size();
	char** argv = (char**)malloc(sizeof(char**) * initial_args.size());
	for(unsigned int i=0; i<initial_args.size(); i++)
	{
		argv[i] = 0;
	}

	for(unsigned int i=0; i<initial_args.size(); i++)
	{
		if(initial_args[i].size())
		{
			argv[argc] = (char*) malloc( initial_args[i].size() + 1);
			strcpy(argv[argc], initial_args[i].c_str());
			argc++;
		}
	}


	// load the modules
	int i = load_items(L, theModules, argc, argv, suppress);

	
	lua_addargs(L, argc, argv);

	
	for(int i=0; i<argc; i++)
	{
		if(argv[i])
		{
			free(argv[i]);
		}
	}
	free(argv);

	int failures = 0;
	for(unsigned int i=0; i<theModules.size(); i++)
	{
		if(!theModules[i].registered)
			failures++;
	}
	
	return failures;
}




// add command line args to the lua state
// adding argc, argv and a table arg
void lua_addargs(lua_State* L, int argc, char** argv)
{
	lua_pushinteger(L, argc);
	lua_setglobal(L, "argc");

	lua_newtable(L);
	for(int i=0; i<argc; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, initial_args[i].c_str());
		lua_settable(L, -3);
	}
	lua_setglobal(L, "argv");
	
	lua_newtable(L);
	int j = 1;
	for(int i=2; i<argc; i++)
	{
		if(argv[i][0])
		{
			lua_pushinteger(L, j);
			lua_pushstring(L, argv[i]);
			lua_settable(L, -3);
			j++;
		}
	}
	lua_setglobal(L, "arg");
}


// get info about the build
static int l_info(lua_State* L)
{
	string result;
	if(lua_gettop(L))
		result.append(lua_tostring(L, 1));

	for(int pos=0; __info[pos]; pos++)
		if(__info[pos] != '\n' || __info[pos+1] != 0)
		{
			result.append(1, __info[pos]);
			if(lua_gettop(L) && __info[pos] == '\n' && __info[pos+1])
					result.append(lua_tostring(L, 1));
		}

	if(lua_gettop(L))
		result.append(lua_tostring(L, 1));

	result.append("\nModules: ");

	for(unsigned int i=0; i<theModules.size(); i++)
	{
		result.append(theModules[i].truename);
		result.append("(r");
		
		std::ostringstream os;
		os << theModules[i].version;
		std::string r_str = os.str(); //retrieve as a string
		
		result.append(r_str);
		result.append(")");

		if( i+1 < theModules.size() )
			result.append(", ");
		
	}
	
	lua_pushstring(L, result.c_str());
	return 1;
}



const char* reference()
{
	return "Use the following reference when citing this code:\n" \
		   /* "*** reference incomplete ***\n" \ */
	       "\"MagLua, a Micromagnetics Programming Environment\". Mercer, Jason I. (2011). Journal. Vol, pages";
}

void print_help()
{
	cout << "MagLua r" << __revi << " by Jason Mercer (c) 2011" << endl;
	cout << endl;
	cout << " MagLua is a micromagnetics programming environment built" << endl;
	cout << " on top of the Lua scripting language." << endl;
	cout << endl;
	cout << "Command Line Arguments:" << endl;
	cout << " -q               Run quietly, omit some startup messages" << endl;
	cout << " --module_path    Print primary module directory" << endl;
#ifdef WIN32
	cout << " --setup mod_dir  Setup startup files in $(APPDATA)\\maglua" << endl;
#else
	cout << " --setup mod_dir  Setup startup files in $(HOME)/.maglua.d" << endl;
#endif
	cout << "                   with <mod_dir> in the list of paths" << endl;
		cout << " -v, --version    Print version" << endl;
	cout << " -h, --help       Show this help" << endl;
	cout << endl;
	cout << reference() << endl;
	
	cout << endl;
}

