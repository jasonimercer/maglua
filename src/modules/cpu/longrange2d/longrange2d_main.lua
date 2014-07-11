-- Longrange2D help func

-- In this file we will add command line help switches
-- controlling tensor cache paths

help_args = help_args or {}

table.insert(help_args, {"",""})
table.insert(help_args, {"LongRange2D Related:",""})
table.insert(help_args, {"--set_longrange2d_cache <dir>", "Set a persistent path for LongRange2D tensor files."})
table.insert(help_args, {"--use_longrange2d_cache <dir>", "Set a single use path for LongRange2D tensor files."})
table.insert(help_args, {"--get_longrange2d_cache", "Get the current path for  LongRange2D tensor files."})

local startup_file = startup_path_dir .. startup_dir_sep .. "longrange2d.lua"

local function setup_persistent_longrange_cache(dir)
    local f = io.open(startup_file, "w")
    if f == nil then
	error("Failed to open `" .. startup_file .. "' for writing. Unable to setup persistent longrange2D cache")
    end

    f:write([[
LongeRange2D = LongRange2D or {}
LongRange2D.cacheDirectory = function() 
				 -- the following should end with a directory separator (unix, mac = /)
				 return "]] .. dir .. startup_dir_sep .. [["
			     end
]])

    f:close()
end

for i,v in ipairs(arg) do
    if v == "--set_longrange2d_cache" then
	local p = arg[i+1]
	if p then
	    table.remove(arg, i+1)
	    setup_persistent_longrange_cache(p)
	end
	table.remove(arg, i)
    end
end

-- load cache dir (may have been updated above)
local f = io.open(startup_file, "r")
if f then
    dofile(startup_file)
    f:close()
end

-- set 1 shot cache dir
for i,v in ipairs(arg) do
    if v == "--use_longrange2d_cache" then
	local p = arg[i+1]
	if p then
	    table.remove(arg, i+1)
	    LongeRange2D = LongRange2D or {}
	    LongRange2D.cacheDirectory = function() 
				 return "]] .. p .. [[" .. startup_dir_sep
			     end
	end
	table.remove(arg, i)
    end
end

-- get cache dir
for i,v in ipairs(arg) do
    if v == "--get_longrange2d_cache" then
	local f = LongRange2D.cacheDirectory or function() return "Not set" end
	print( f() )
	shutdown_now = true -- tell the bootloader that things should stop now before we try to start a script 
    end
end
