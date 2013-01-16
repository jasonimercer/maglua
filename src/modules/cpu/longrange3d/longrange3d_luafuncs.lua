-- LongRange3D support functions

local MODNAME = "LongRange3D"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

	
local function matStrings(lr)
	local filled = {}
	local empty = {}
	for k,mat in pairs({"XX", "XY", "XZ", "YY", "YZ", "ZZ"}) do
		local a = lr:getArray(mat)
		
		for z=1,a:nz() do
			local sliceSum = 0
			local t = {}
			for y=1,a:ny() do
				local r = {}
				for x=1,a:nx() do
					sliceSum = sliceSum + math.abs( a:get(x,y,z) )
					r[x] = string.format("%18.16e", a:get(x,y,z))
				end
				t[y] = table.concat(r, ", ")
			end
			
			if sliceSum ~= 0 then
				table.insert(filled, string.format("%s[%d] = [[\n%s\n]]", mat, z-1, table.concat(t, ",\n")))
			else
				table.insert(empty, string.format("%s[%d]", mat, z-1)) 
			end
		end
	end
	
	local e = ""
	local radix = 5
	local r = math.floor(table.maxn(empty) / radix)
	
	local zeros = {}
	for i=1,radix do
		zeros[i] = 0
	end
	zeros = table.concat(zeros, ",")
	
	for i=1,r do
		local j = i-1
		for k=1,radix-1 do
			e = e .. empty[j*radix+k] .. ", "
		end
		e = e .. empty[j*radix+radix] .. " = " .. zeros .. "\n"
	end
	local n = 0
	local foo = {}
	local bar = {}
	for i=r*radix,table.maxn(empty) do
		table.insert(foo, empty[i])
		table.insert(bar, 0)
	end
	
	if table.maxn(foo) > 0 then
		e = e .. table.concat(foo, ", ") .. " = " .. table.concat(bar, ",") .. "\n"
	end
	
	if string.len(e) > 0 then
		e = "-- The following slices are identically equal to zero.\n-- Functions below will expand them to arrays that will return zero\n" .. e
	end
	
	return 
	[[
XX,XY,XZ={},{},{}
   YY,YZ=   {},{}
      ZZ=      {}
	  
]] .. table.concat(filled, "\n\n") .. "\n\n" .. e
end
	
local function trace()
	local t = {}
	local s = debug.traceback("",3)
	for k, v in string.gmatch(s, "(.-)\n") do
		table.insert(t, "-- " .. k)
	end
	table.remove(t, 1)
	table.remove(t, 1)
	table.remove(t)

	return table.concat(t, "\n") .. "\n" 
end

local function setMatrix(lr, ab, x,y,z, value) --expecting base 0
	if type(x) == "table" then
		return setMatrix(lr, ab, x[1], x[2], x[3], y)
	end
	
	local a = lr:tensorArray(ab)
	a:set(x+1,y+1,z+1, value)
	
	lr:setNewDataRequired(false)
	lr:setCompileRequired(true)
end

local function saveTensors(lr, filename, notes)
	if notes then
		notes = "-- " .. string.gsub(notes, "\n", "\n-- ") .. "\n"
	else
		notes = "\n"
	end
	local f = io.open(filename, "w")
	if f == nil then
		error("Failed to open `" .. filename .. "' for writing")
	end
	
	f:write([[-- This is a custom 3D tensor file created with MagLua-r]] .. version() .. [[.
-- Use lr3d:loadTensors(filename), where lr3d is a LongRange3D object, to load these tensors.
-- The matrices below represent slices in the XY plane, the indices on the AB tables represent Z offsets.

-- This data was created at:
]] .. trace() .. [[
	
	
]] .. notes .. 
[[

local nx, ny, nz = ]] .. table.concat({lr:nx(), lr:ny(), lr:nz()}, ", ") .. [[ -- dimensions of saved tensor

function loadTensor(lr3d)
	if nx ~= lr3d:nx() or ny ~= lr3d:ny() or nz ~= lr3d:nz() then
		error("Tensor size mismatch")
	end
	
	parse() --convert the data below from strings to tables of numbers
	
	local mat = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"}
	local dat = { XX ,  XY ,  XZ ,  YY ,  YZ ,  ZZ }
	for i=1,6 do
		local ab = dat[i]
		
		for z=1,nz do
			for y=1,ny do
				for x=1,nx do
					lr3d:setMatrix(mat[i], {x-1,y-1,z-1}, ab[z-1][y][x])
				end
			end
		end
	end
end

-- Matrix columns step in the lattice X direction (A vector), rows step in the Y direction (B vector)

]]

.. matStrings(lr) ..

[[

function parse()
	local function map(f, t)
		for k,v in pairs(t) do
			t[k] = f(v)
		end
		return t
	end
	
	local function tokenizeNumbers(line)
		local t = {}
		for w in string.gmatch(line, "[^,]+") do
			table.insert(t, tonumber(w))
		end
		return t
	end

	local function tokenizeLines(lines)
		-- strip empty lines
		lines = string.gsub(lines, "^%s*\n*", "")
		lines = string.gsub(lines, "\n\n+", "\n")
		
		local t = {}
		for w in string.gmatch(lines, "(.-)\n" ) do
			table.insert(t, w)
		end
		return map(tokenizeNumbers, t)
	end

	local function parseMatrix(M)
		if M == 0 then
			-- returns a 2D table that always returns zero
			local tz, ttz = {}, {}
			setmetatable(tz,  {__index = function() return  0 end})
			setmetatable(ttz, {__index = function() return tz end})
			return ttz
		end
		
		return tokenizeLines(M)
	end

	
	XX = map(parseMatrix, XX)
	XY = map(parseMatrix, XY)
	XZ = map(parseMatrix, XZ)

	YY = map(parseMatrix, YY)
	YZ = map(parseMatrix, YZ)

	ZZ = map(parseMatrix, ZZ)
end
]])
	
	f:close()
end

local function loadTensors(lr, filename)
	dofile(filename)
	loadTensor(lr)
end

t.saveTensors = saveTensors
t.loadTensors = loadTensors
t.setMatrix  = setMatrix

local help = MODTAB.help

MODTAB.help = function(x)
	if x == saveTensors then
		return
		"Convenience method to save the tensors in a lua readable file",
		"2 Strings: The filename used for writing the tensor file, traditionally ending with a .lua extension. Optional note to put in the file.",
		""
	end
	
	if x == loadTensors then
		return
		"Convenience method to load interaction tensors from a file",
		"1 String: The filename of the tensor file.",
		""
	end
	if x == setMatrix then
		return
		"Set a tensor element",
		"1 String, 3 Numbers or 1 Table of 3 Numbers, 1 Number: Tensor name, base 0 element coordinate in the tensor, new value",
		""
	end
	
	if x == nil then
		return help()
	end
	return help(x)
end

