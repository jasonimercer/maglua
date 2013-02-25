-- LongRange2D
local MODNAME = "LongRange2D"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local function getMatrix(lr2d, dest, src, x, y, ab)
	local q = lr2d:tensorArray(dest, src, ab)
	return q:get(x,y)
end



t.getMatrix = getMatrix

local help = MODTAB.help

MODTAB.help2 = function(x)
	if x == getMatrix then
		return
			"Get an element of the interaction tensor",
			"4 Integers, 1 String: destination layer (base 1), source layer (base 1), x and y offset, tensor name: XX, XY, XZ, etc",
			"1 Number: Tensor element"
	end

	if x == nil then
		return help()
	end
	return help(x)
end

