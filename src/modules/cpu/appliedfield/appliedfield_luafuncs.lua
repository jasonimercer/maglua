-- AppliedField 

local MODNAME = "AppliedField"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local function set(a, x,y,z)
	if type(x) == "table" then
		z = x[3] or 0
		y = x[2] or 0
		x = x[1] or 0
	else
		x = x or 0
		y = y or 0
		z = z or 0
	end
	
	a:setX(x)
	a:setY(y)
	a:setZ(z)
end

	
local function add(a, x,y,z)
	if type(x) == "table" then
		z = x[3] or 0
		y = x[2] or 0
		x = x[1] or 0
	end
	
	a:setX(a:x() + x)
	a:setY(a:y() + y)
	a:setZ(a:z() + z)
end

	
local function get(a)
	return a:x(), a:y(), a:z()
end
 
local function toTable(a)
    return {a:x(), a:y(), a:z()}
end


local function apply(a, ss)
	if type(ss) == "table" then
		for k,s in pairs(ss) do
			a:apply(s)
		end
		return
	end
	ss:ensureSlotExists(a:slotName())
	local name = a:slotName()
	local scale = a:scale()
	ss:fieldArrayX(name):setAll(a:x() * scale)
	ss:fieldArrayY(name):setAll(a:y() * scale)
	ss:fieldArrayZ(name):setAll(a:z() * scale)
	ss:setSlotUsed(name)
end

t.set = set
t.get = get
t.add = add
t.toTable = toTable
t.apply = apply

local help = MODTAB.help

MODTAB.help = function(x)
	if x == set then
		return 
			"Set the direction and strength of the Applied Field",
			"1 *3Vector*: The *3Vector* defines the strength and direction of the applied field",
			""
	end
		
	if x == add then
		return
			"Add the direction and strength of the Applied Field",
			"1 *3Vector*: The *3Vector* defines the strength and direction of the applied field addition",
			""
	end

	if x == get then
		return 
		"Get the direction and strength of the Applied Field",
		"",
		"3 numbers: The x, y and z components of the field"
	end
		
	if x == toTable then
		return 
		"Get the direction and strength of the Applied Field",
		"",
		"1 Table of 3 Numbers: The x, y and z components of the field"
	end
		
	if x == nil then
		return help()
	end
	return help(x)
end

