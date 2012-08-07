-- bootstrap   
-- 
-- This is a boot loader for the rest of the program
-- this file is encoded into a C string

-- OS specific file locations/cmds
local module_path_dir
local module_path_file
local module_short_dir
local module_dir_sep
local mkdir_cmd
if ENV == "WIN32" then
	module_path_dir = os.getenv("APPDATA") .. "\\maglua"
	module_path_file= os.getenv("APPDATA") .. "\\maglua\\module_path.lua"
	mkdir_cmd = function(x) return "mkdir \"" .. x .. "\"" end
	module_short_dir = "$(APPDATA)\\maglua"
	module_dir_sep = "\\"
else
	module_path_dir = os.getenv("HOME") .. "/.maglua.d"
	module_path_file= os.getenv("HOME") .. "/.maglua.d/module_path.lua"
	mkdir_cmd = function(x) return "mkdir -p \"" .. x .. "\"" end
	module_short_dir = "$(HOME)/.maglua.d"
	module_dir_sep = "/"
end
ENV = nil

local be_quiet = nil
local print_module_path = nil
local print_version = false
local setup_module_path = nil
local print_help = false
local print_module_path_file = nil


-- trim info about bootstrap from error messages
debug["trim_error"] = function(msg)
	local x = 
[[
%s*%[C%]%: in function .dofile.%s*
%s*%[string "%-%-%s*bootstrap]]

	local a, b, m = string.find(msg, [[^(.*)]] .. x)
	if a then 
		return m 
	end
	return msg
end




for k,v in pairs(arg) do
	if v == "--module_path" and sub_process == nil then
		print_module_path = arg[k+1] or true
	end
	
	if v == "--use_module_file" then
		module_path_file = arg[k+1]
	end
	
	if v == "--module_file" and sub_process == nil  then
		print_module_path_file = arg[k+1] or true
	end
	
-- 	Moving this test to after dofile so that -q can be included removed by module_path_file
-- 	if v == "-q" then
-- 		be_quiet = k
-- 	end
	
	if (v == "-v" or v == "--version") and sub_process == nil then
		print_version = true
	end
	
	if v == "--setup" and sub_process == nil then
		setup_module_path = arg[k+1]
		--need to strip quotes from it if they exist
		local a, b, c = string.find(setup_module_path, "^%s*%'(.*)%'%s*$")
		if a then
			setup_module_path = c
		end
		local a, b, c = string.find(setup_module_path, "^%s*%\"(.*)\"%s*$")
		if a then
			setup_module_path = c
		end
	end
	
	if (v == "-h" or v == "--help")  and sub_process == nil then
		print_help = true
	end
end



function reference()
	return "Publications with results derived from MagLua must co-author the following:\n" ..
			"   Jason I. Mercer, Department of Computer Science,\n   Memorial University of Newfoundland. jason.mercer@mun.ca"
--	return "Use the following reference when citing this code:\n" ..
--		   [["MagLua, a Micromagnetics Programming Environment". Mercer, Jason I. (2012). Journal. Vol, pages]]
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
	if setup_module_path then
		print("Creating `module_path.lua' file in `" .. module_path_dir .. "'")
		print("Setting path `" .. setup_module_path .. "'")

		os.execute(mkdir_cmd(module_path_dir))
		local f = io.open(module_path_file, "w")
		
		f:write([[-- Modules in the following directories can be loaded
module_path = {}
module_path["common"] = "]] .. escape(setup_module_path .. module_dir_sep .. "common") .. [["
module_path["cpu"]    = "]] .. escape(setup_module_path .. module_dir_sep .. "cpu"   ) .. [["
module_path["cuda"]   = "]] .. escape(setup_module_path .. module_dir_sep .. "cuda"  ) .. [["
module_path["cuda32"] = "]] .. escape(setup_module_path .. module_dir_sep .. "cuda32") .. [["
module_path["extra"]  = "]] .. escape(setup_module_path .. module_dir_sep .. "extra" ) .. [["
	
-- Modules in the following categories will be loaded
use_modules = {"common", "cpu", "extra"}

-- This startup file can be highly customized.
-- You can interact with command line arguments via arg,
-- the help_args can also be changed.
-- Example, always running quietly(-q) unless +q:
]].."--[[\n"..[[
help_args["-q"] = nil
help_args["+q"] = "Print startup messages and list messages"
local foundq = nil
for k,v in pairs(arg) do
	if v == "+q" then
		foundq = k
	end
end
if foundq then
	table.remove(arg, foundq)
else
	table.insert(arg, 1, "-q")
end
]].."--]]\n"..[[
-- Example, custom arg[] parsing to build the use_modules table:
]].."--[[\n"..[[
if arg[2] == "gpu" then
	table.remove(arg, 2)
	use_modules = {"common", "cuda", "cuda32", "extra"} 
else
	use_modules = {"common", "cpu", "extra"} 
end
]].."--]]\n"..[[

-- You can also maintain different module directories using the "version()" function: "common-r" .. version()
--  Be aware that if you run maglua --setup this_path again, this file will be overwritten by the default.
]])
		f:close()
		return false
	end

	if print_module_path then
		dofile(module_path_file)
		local mp = module_path[print_module_path]
		if mp then
			print(mp)
			return false
		end

		local cats = {}
		for k,v in pairs(module_path) do
			table.insert(cats, k)
		end
		
		io.stderr:write("Failed to lookup module category\n")
		error("Must specify a module category (" .. table.concat(cats, ", ") .. ")", 0)
	end

	if print_module_path_file then
		print(module_path_file)
		return false
	end

	if print_version then
		print("MagLua-r" .. version())
		return false -- stop here
	end
end 




help_args = {}

help_args["-q"] =                       "Run quietly, omit some startup messages"
help_args["--setup path"] =             "Setup startup files in " .. module_short_dir
help_args["--module_file"] =            "Print the name of the file that manages modules"
help_args["--use_module_file <file>"] = "Use the given file to manage modules"
help_args["--module_path <category>"] = "Print module directory for <category> module types"
help_args["-v, --version"] =            "Print version"
help_args["-h, --help"] =               "Show this help"

-- get the module path
dofile(module_path_file)

if print_help then
	print("MagLua-r" .. version()..  " by Jason Mercer (c) 2012\n")
	print(
[[ MagLua is a micromagnetics programming environment built
 on top of the Lua scripting language.

 Command Line Arguments:
]])

	local len = 0
	for k,v in pairs(help_args) do
		if string.len(k) > len then
			len = string.len(k)
		end
	end

	for k,v in pairs(help_args) do
		print(string.format("  %-" .. len .. "s  %s", k,v))
	end
	print()

	print(reference())
	return false
end



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


if use_modules == nil then
	error("use_modules not defined in `module_path.lua'")
end

-- build a big flat list of modules to load
local mod = {}

for k1,v1 in pairs(use_modules) do
	for k2,v2 in pairs(getModuleDirectory(module_path[v1])) do
		table.insert(mod, {path=v2})
	end
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

for k,v in pairs(mod) do
	for a,b in pairs(v.result) do
		mod[k][a] = b
	end
end

function make_modules()
	local m = mod
	return function()
		return m
	end
end
modules = make_modules()


local function e(x)
	io.stderr:write(x .. "\n")
end

if be_quiet == nil then
	e("This evaluation version of MagLua is for academic, non-commercial use only")
	e("MagLua-r" .. version() .. " by Jason Mercer (c) 2012\n")
	e( reference() .. "\n")
	e("Modules:")
	local t = {}
	for k,v in pairs(modules()) do
		table.insert(t, v.name .. "(" .. v.version .. ")")
	end
	table.sort(t)
	e(table.concat(t, ", "))
end


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
		e("Please supply a MagLua script (*.lua)")
		return false
	end

	-- need to shift indexes so that script is at [0]
	local a = {}
	for k,v in pairs(arg) do
		a[k - first_script_index] = v
	end
	arg = a

	dofile(a[0])
else
	sub_process = nil
end




