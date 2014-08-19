-- bootstrap   
-- 
-- This is a boot loader for the rest of the program
-- this file is encoded into a C string

-- OS specific file locations/cmds
-- for now OS is only Linux
local module_path_dir
local startup_path_file
local startup_short_dir
startup_dir_sep = "" -- making this global, setting to nil near the end
local mkdir_cmd
if ENV == "WIN32" then
    startup_path_dir = os.getenv("APPDATA") .. "\\maglua"
    startup_path_file= os.getenv("APPDATA") .. "\\maglua\\startup.lua"
    mkdir_cmd = function(x) return "mkdir \"" .. x .. "\"" end
    startup_short_dir = "$(APPDATA)\\maglua"
    startup_dir_sep = "\\"
else
    startup_path_dir = os.getenv("HOME") .. "/.maglua.d"
    startup_path_file= os.getenv("HOME") .. "/.maglua.d/startup.lua"
    mkdir_cmd = function(x) return "mkdir -p \"" .. x .. "\"" end
    startup_short_dir = "$(HOME)/.maglua.d"
    startup_dir_sep = "/"
end
ENV = nil

local be_quiet = nil
local print_module_path = nil
local print_version = false
local setup_startup_path = nil
local print_help = false
local print_startup_path_file = nil
local write_documentation = nil
local cmd_line_max_memory = nil
local do_error_trim = false
local custom_load_module = {}
local skip_startup_file = false
delayed_execute = {}

-- trim info about bootstrap from error messages
debug["trim_error"] = 
    function(msg)
	if do_error_trim == false then
	    return msg
	end
	msg = msg or "error"
	local a, b, preStack, Stack, postStack = string.find(msg, "(.-)(stack traceback%:%s*)(.*)")
	
	if a == nil then
	    return msg --don't know how to deal with it
	end
	
	-- need to trim error function from postStack
	local a, b, c = string.find(postStack, "%s*%[C%]%: in function %'error%'%s+(.*)")
	if a then
	    postStack = c
	end
	
	-- need to trim bootstrap scope
	local a, b, c = string.find(postStack, "(.*)%s*%[C%]%: in function .dofile_original.%s.*")
	if a then
	    postStack = c
	end
	
	return preStack .. Stack .. postStack
    end




for k,v in pairs(arg) do
    if v == "--module_path" and sub_process == nil then
	print_module_path = arg[k+1] or true
    end
    
    if v == "--use_startup_file" then
	startup_path_file = arg[k+1]
    end
    
    if v == "--load_module" then
	local mod_name = arg[k+1]

	local a,b = string.find(mod_name, "/")
	if a == nil then
	    -- we will treat it as a local file and add "./" to make dlopen happy
	    mod_name = "./" .. mod_name
	end

	table.insert(custom_load_module, mod_name)
    end

    if v == "--skip_startup_file" then
	skip_startup_file = true
    end



    if v == "--startup_file" and sub_process == nil  then
	print_startup_path_file = arg[k+1] or true
    end
    
    if v == "--write_docs" and sub_process == nil then
	write_documentation = k --arg[k+1] or true
    end

    -- 	Moving this test to after dofile so that -q can be included removed by startup_path_file
    -- 	if v == "-q" then
    -- 		be_quiet = k
    -- 	end
    
    if (v == "-v" or v == "--version") and sub_process == nil then
	print_version = true
    end
    
    if v == "--setup" and sub_process == nil then
	setup_startup_path = arg[k+1]
	--need to strip quotes from it if they exist
	local a, b, c = string.find(setup_startup_path, "^%s*%'(.*)%'%s*$")
	if a then
	    setup_startup_path = c
	end
	local a, b, c = string.find(setup_startup_path, "^%s*%\"(.*)\"%s*$")
	if a then
	    setup_startup_path = c
	end
    end
    
    if v == "--maxMemory" then
	local mem_err = "--maxMemory requires a value and unit. Example: --maxMemory 1024MB or --maxMemory 1.3GB"
	local units = {B=1024^0, KB=1024^2, MB=1024^2, GB=1024^3, TB=1024^4}
	arg[k+1] = arg[k+1] or ""
	local a,b,c,d = string.find(arg[k+1], "^%s*(%d+)(%a*)%s*$")
	if d then
	    d = string.upper(d)
	    d = units[d] or error(mem_err)
	    cmd_line_max_memory = tonumber(c) * d
	else
	    error(mem_err)
	end
    end
    
    
    if (v == "-h" or v == "--help")  and sub_process == nil then
	print_help = true
    end
end



function reference()
    return nil
    --	return "Publications with results derived from MagLua must co-author the following:\n" ..
    --			"   Jason I. Mercer, Department of Computer Science,\n   Memorial University of Newfoundland. jason.mercer@mun.ca"
    --	return "Use the following reference when citing this code:\n" ..
    --		   [["MagLua, a Micromagnetics Programming Environment". Mercer, Jason I. (2014). Journal. Vol, pages]]
end

local function make_version()
    local v = __version
    return function()
	       return v
	   end
end
version = make_version()
__version = nil


local function make_info()
    local v = __info
    return function(p)
	       p = p or ""
	       return p .. string.gsub(v, "\n", "\n" .. p)
	   end
end
info = make_info()
__info = nil

function escape(line)
    return string.gsub(line, "\\", "\\\\")
end

if not print_help then
    if setup_startup_path then
	print("Creating `startup.lua' file in `" .. startup_path_dir .. "'")
	print("Setting path `" .. setup_startup_path .. "'")

	os.execute(mkdir_cmd(startup_path_dir))
	local f = io.open(startup_path_file, "w")
	
	f:write([[-- Modules in the following directories can be loaded
module_path = {}
module_path["common"] = "]] .. escape(setup_startup_path .. startup_dir_sep .. "common") .. [["
module_path["cpu"]    = "]] .. escape(setup_startup_path .. startup_dir_sep .. "cpu"   ) .. [["
module_path["cuda"]   = "]] .. escape(setup_startup_path .. startup_dir_sep .. "cuda"  ) .. [["
module_path["cuda32"] = "]] .. escape(setup_startup_path .. startup_dir_sep .. "cuda32") .. [["
module_path["extra"]  = "]] .. escape(setup_startup_path .. startup_dir_sep .. "extra" ) .. [["

-- Modules in the following categories will be loaded
use_modules = {"common", "cpu", "extra"}

-- This startup file can be highly customized.
-- You can interact with command line arguments via arg,
-- the help_args can also be changed.
-- Example, always running quietly(-q) unless +q:
]].."--[[\n"..[[
for i,v in ipairs(help_args) do
    if v[1] == "-q" then
	table.remove(help_args, i) -- delete help for -q
    end
end

-- add new help
table.insert(help_args, {"+q", "Print startup messages and list messages"})
	     
local foundq = nil
for k,v in pairs(arg) do
    if v == "+q" then
	foundq = k
    end
end
if foundq then
    table.remove(arg, foundq)
else
    table.insert(arg, 1, "-q") -- default: add be quiet
end
]].."--]]\n"..[[
-- Example, custom arg[] parsing to build the use_modules table:
]].."--[[\n"..[[
if arg[2] == "gpu" then
    table.remove(arg, 2)
    use_modules = {"common", "cuda", "extra"} 
else
    use_modules = {"common", "cpu", "extra"} 
end
]].."--]]\n"..[[
	
-- You can also maintain different module directories using the "version()" function: "common-r" .. version()
-- Be aware that if you run maglua --setup this_path again, this file will be overwritten by the default.
	
-- You can set memory limits for each maglua process by defining one of the following values
-- Less than 200MB may make initialization fail if many modules are required. If these values are not
-- set then the system defaults are used. 
-- maxMemoryMB = 512 -- limit system memory use to 512 MB
-- maxMemoryGB = 6   -- limit system memory use to   6 GB
]])
f:close()
return false
end

if print_module_path then
    dofile(startup_path_file)
    local mp = module_path[print_module_path]
    if mp then
	print(mp)
	return false
    end
    
    local cats = {}
    for k,v in pairs(startup_path) do
	table.insert(cats, k)
    end
    
    io.stderr:write("Failed to lookup module category\n")
    error("Must specify a module category (" .. table.concat(cats, ", ") .. ")", 0)
end

if print_startup_path_file then
    print(startup_path_file)
    return false -- stop here
end

if print_version then
    print("MagLua-r" .. version())
    return false -- stop here
end
end 



-- a module may have inserted help args
-- so we'll default to a table
help_args = help_args or {}

table.insert(help_args,{"Common Options",""})
table.insert(help_args,{"-h, --help",               "Show this help"})
table.insert(help_args,{"-v, --version",            "Print version"})
table.insert(help_args,{"-q",                       "Run quietly, omit some startup messages"})

table.insert(help_args,{"",""}) -- intentionally blank

table.insert(help_args,{"Advanced Options",""})
table.insert(help_args,{"--setup path",             "Setup startup files in " .. startup_short_dir})
table.insert(help_args,{"--module_path <category>", "Print module directory for <category> module types"})
table.insert(help_args,{"--startup_file",           "Print the name of the file that manages modules"})
table.insert(help_args,{"--use_startup_file <file>","Use the given file to manage modules"})
table.insert(help_args,{"--skip_startup_file",      "Skip the startup file without giving error messages"})
table.insert(help_args,{"--load_module <file>", "Attempt to load the following module. This option can be repeated."})
table.insert(help_args,{"--write_docs [file]",      "Write HTML documentation to given file or stdout"})
table.insert(help_args,{"--maxMemory value(B|KB|MB|GB)","Set the maximum amount of system memory that this process can allocate."})

-- get the module path
--this file does not exist on a clean install
if skip_startup_file == false then
    local mod_test = io.open(startup_path_file, "r") 
    if mod_test then 
	dofile(startup_path_file)
	mod_test:close()
    end
end

-- dealing with memory limits as defined in startup_path_file or command line
if cmd_line_max_memory then
    local a = getrlimit(RLIMIT.AS)
    a[1] = cmd_line_max_memory
    setrlimit(RLIMIT.AS, a)
else
    local maxMemoryBytes = -1
    
    if maxMemoryMB ~= nil then
	maxMemoryBytes = maxMemoryMB * 1024 * 1024
    end
    if maxMemoryGB ~= nil then
	maxMemoryBytes = maxMemoryGB * 1024 * 1024 * 1024
    end
    if maxMemoryBytes > 0 then
	local a = getrlimit(RLIMIT.AS)
	a[1] = maxMemoryBytes
	setrlimit(RLIMIT.AS, a)
    end
end

-- clear memory functions and data
maxMemoryBytes = nil
maxMemoryMB = nil
maxMemoryGB = nil

setrlimit = nil
getrlimit = nil
RLIMIT = nil




-- test for "-q"
be_quiet = nil
for k,v in pairs(arg) do
    if v == "-q" then
	be_quiet = k
    end
end
if be_quiet then
    table.remove(arg, be_quiet)
end


if use_modules == nil and skip_startup_file == false then
    error("use_modules not defined in `startup.lua', do you need to --setup?")
end

use_modules = use_modules or {}

-- build a big flat list of modules to load
local mod = {}

for k1,v1 in pairs(use_modules) do
    for k2,v2 in pairs(getModulesInDirectory(module_path[v1])) do
	table.insert(mod, {path=v2})
    end
end

for k,v in pairs(custom_load_module) do
    table.insert(mod, {path=v})
end

-- iterate through modules until they're all loaded
-- or we can't load anymore
local keep_trying = true
local last_error
local num_loaded
while keep_trying do
    num_loaded = 0
    last_error = {}
    
    for k,v in pairs(mod) do
	if v.result == nil then
	    -- print(v.path)
	    -- print("loadModule(" .. v.path .. ")")
	    local t = loadModule(v.path)
	    if t.error then
		table.insert(last_error, t.error)
	    else
		num_loaded = num_loaded + 1
		v.result = t
	    end
	end
    end
    
    if num_loaded == 0 then
	keep_trying = false
	
	if table.maxn(last_error) > 0 then
	    io.stderr:write("Failed to load module\n")
	    error(table.concat(last_error, "\n"), 0)
	end
    end
end

-- shutdown_now is a flag that can be set in 
-- a module's "main". This can be used, for example,
-- if a module has printed something related to a command
-- line request. Perhaps version data or path data.
if shutdown_now then
    return false
end

for k,v in pairs(mod) do
    for a,b in pairs(v.result) do
	mod[k][a] = b
    end
end

-- creating a closure for modules
local function make_modules()
    local m = mod
    return function()
	       return m
	   end
end
modules = make_modules()


if print_help then
    print("MagLua-r" .. version()..  " by Jason I. Mercer (c) 2014\n")
    print(
[[ MagLua is a micromagnetics programming environment built
 on top of the Lua scripting language.
		
 Command Line Arguments:
]])
    
    local len = 0
    for i,v in ipairs(help_args) do
	if string.len(v[1]) > len then
	    len = string.len(v[1])
	end
    end
    
    for i,v in ipairs(help_args) do
	print(string.format("  %-" .. len .. "s  %s", v[1],v[2]))
    end
    print()
    
    if reference() then
	print(reference())
    end
    return false
end






local function e(x)
    io.stderr:write(x .. "\n")
end

if write_documentation ~= nil then
    local f = nil
    if arg[write_documentation] == nil then
	f = io.stdout
	write_documentation = "stdout"
    else
	f = io.open(arg[write_documentation+1], "w")
	if f == nil then
	    error("Failed to open `" .. arg[write_documentation+1] .. "' for writing")
	end
    end
    
    dofile("maglua://Help.lua")
    
    local optional_doc_data = {}
    for i=write_documentation+2, table.maxn(arg), 2 do
	optional_doc_data[arg[i]] = arg[i+1]
    end
    
    write_help(f, optional_doc_data)
    
    if f ~= io.stdout then
	print("Documentation written to `" .. arg[write_documentation+1] .. "'")
	f:close()
    end
    
    return false --the is an end point for the script
end


if be_quiet == nil then
    -- e("This evaluation version of MagLua is for academic, non-commercial use only")
    e("MagLua-r" .. version() .. " by Jason Mercer (c) 2014\n")
    if reference() then
	e( reference() .. "\n")
    end
    e("Modules:")
    local t = {}
    for k,v in pairs(modules()) do
	table.insert(t, v.name .. "(" .. v.version .. ")")
    end
    table.sort(t)
    e(table.concat(t, ", "))
end

-- some mains may have put functions in delayed_execute so we'll do them now
-- the reason they're done now is that all modules are loaded while
-- in main they may not have been loaded
for k,v in pairs(delayed_execute) do
    v()
end

-- giving shutdown_now another change. 
if shutdown_now then
    return false
end


-- cleaning up the scope
-- removing variables that are not needed
delayed_execute = nil
startup_dir_sep = nil
do_error_trim = true
startup_path_dir = nil
escape = nil
getModulesInDirectory = nil
use_modules = nil

if sub_process == nil then
    -- find first script in args
    local first_script_index = nil
    
    for i=1,table.maxn(arg) do
	if arg[i] and first_script_index == nil then
	    local a, b = string.find(string.lower(arg[i]), ".*%.lua$")
	    if a then
		first_script_index = i
	    end
	end
    end
    
    if first_script_index == nil then
	if interactive then
	    interactive("", {message=false, header=false})
	    print() -- for the newline that ctrl+d doesn't give
	else
	    e("Please supply a MagLua script (*.lua)")
	end
	return false
    end
    
    -- need to shift indexes so that script is at [0]
    local a = {}
    for k,v in pairs(arg) do
	a[k - first_script_index] = v
    end
    arg = a
    a = nil

    dofile(arg[0])
else
    sub_process = nil
end




