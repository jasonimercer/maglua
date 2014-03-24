local help = Array.help

Array.help = function(x)
	if x == nil then
		return "This table has functions related to instantiating annd operating on high performance arrays."
	end
	if x == Array.DoubleComplexWorkSpace then
		return "Get a Double Precision Complex workspace for temporary operations",
		       "3 Integers, 1 String: The size in the x, y and z directions as well as a name. The name allows multiple workspaces who share the same name to potentially share the same memory",
			   "1 Double Compex Array"
	end

	if x == Array.DoubleWorkSpace then
		return "Get a Double Precision workspace for temporary operations",
		       "3 Integers, 1 String: The size in the x, y and z directions as well as a name. The name allows multiple workspaces who share the same name to potentially share the same memory",
			   "1 Double Precision Array"
	end

	if x == Array.WorkSpaceInfo then
		return "Get a table with information about the workspace buffers",
		"",
		"1 Table of tables: Each table has two keys: size and hash giving the number of bytes and the hashed form of the name for each registered workspace"
	end


	if help then
		if x == nil then
			return help()
		end
		return help(x)
	end
end


-- the workspace needs to get registered and then unregistered on application
-- shutdown. We're doing this with an object that will call specialized code
-- on it's garbage collection. NOTE: This doesn't work with Lua 5.1, it does with 5.2

Array._registerws()

Array._workspace_object = {}

local unregws = Array._unregisterws
local mt = {__gc = function() unregws() end,
			__tostring = function() return "This object will call custom cleanup code on it's garbage collection" end}

setmetatable(Array._workspace_object, mt)

-- now that we have the workspace registered and a way to unregister it, we'll 
-- remove the functions from the scope to prevent problems
Array._unregisterws = nil
Array._registerws = nil

