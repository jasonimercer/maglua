#ifndef SETUP_CODE
#define SETUP_CODE

#ifndef WIN32
static const char* setup_lua_code = 

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

#error Need code for Windows setup

#endif

#endif //SETUP_CODE
