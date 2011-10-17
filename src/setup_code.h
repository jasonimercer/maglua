#ifndef SETUP_CODE
#define SETUP_CODE

static const char* setup_lua_code = 
#ifndef WIN32
"function setup(mod_path)\n"\
"	print(\"Creating default startup files in $(HOME)/.maglua.d\")\n"\
"	print(\"adding path `\" .. mod_path .. \"'\")\n"\
"	local home = os.getenv(\"HOME\")\n"\
"	os.execute(\"mkdir -p \" .. home .. \"/.maglua.d/\")\n"\
"	f = io.open(home .. \"/.maglua.d/module_path.lua\", \"w\")\n"\
"	f:write(\"-- Modules in the following directories will be loaded\\n\")\n"\
"	f:write(\"module_path = {\\\"\" .. mod_path .. \"\\\"}\\n\")\n"\
"end\n";
#else
"function setup(mod_path)\n"\
"   mod_path = string.gsub(mod_path, \"\\\\\", \"\\\\\\\\\")\n"\
"	print(\"Creating default startup files in $(APPDATA)\\\\maglua\")\n"\
"	print(\"adding path `\" .. mod_path .. \"'\")\n"\
"	local home = os.getenv(\"APPDATA\")\n"\
"	os.execute(\"mkdir \\\"\" .. home .. \"\\\\maglua\")\n"\
"	f = io.open(home .. \"\\\\maglua\\\\module_path.lua\", \"w\")\n"\
"	f:write(\"-- Modules in the following directories will be loaded\\n\")\n"\
"	f:write(\"module_path = {\\\"\" .. mod_path .. \"\\\"}\\n\")\n"\
"end\n";
#endif

#endif //SETUP_CODE
