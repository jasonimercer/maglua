-- AppliedField.Heterogeneous

local MODNAME = "AppliedField.Heterogeneous"
local MODTAB = AppliedField.Heterogeneous
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local set = nil
local function set(a, p, h)
	if type(p) ~= "table" then
		error("first argument needs to be a table")
	end
	if type(h) ~= "table" then
		error("second argument needs to be a table")
	end
	
	a:fieldArrayX():set(p, h[1])
	a:fieldArrayY():set(p, h[2])
	a:fieldArrayZ():set(p, h[3])
end

local function get(a, p)
	if type(p) ~= "table" then
		error("first argument needs to be a table")
	end
	
	local x = a:fieldArrayX():get(p)
	local y = a:fieldArrayY():get(p)
	local z = a:fieldArrayZ():get(p)
	
	return {x,y,z}
end
 

t.set = set
t.get = get

local help = MODTAB.help

MODTAB.help = function(x)
	if x == set then
		return 
			"Set the direction and strength of the Heterogeneous Applied Field at a position",
			"2 Tables: The first table is the position, the second is the local field",
			""
	end
		
	if x == get then
		return 
		"Get the direction and strength of the Heterogeneous Applied Field at a positon",
		"1 Tables: The table defines the position",
		"1 Table of 3 numbers: The x, y and z components of the field at the given position"
	end
		
	if x == nil then
		return help()
	end
	return help(x)
end

