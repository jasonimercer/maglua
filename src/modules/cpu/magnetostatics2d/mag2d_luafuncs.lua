-- Magnetostatics2D

local MODNAME = "Magnetostatics2D"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

-- at first the internal data is nil. If it is it
-- needs to be setup for mag2d
local function initializeInternalData(mag, t)
	if t ~= nil then
		return t --it's OK
	end
	t = {}
	t.ABC = {}
	t.grainSize = {}
	t.truncation = 25
	for i=1,mag:nz() do
		t.ABC[i]       = {{1,0,0}, {0,1,0}, {0,0,1}}
		t.grainSize[i] =  {1,1,1}
	end
	return t
end
	

local function isNumber(x) 
	return type(x) == "number"
end

local function isTable(x)
	return type(x) == "table"
end

local function getMatrix(mag, dest, src, ab, x, y)
	mag:makeData() -- if needed
	if dest == nil then	error("Destination Layer is nil", 2) end
	if  src == nil then	error("Source Layer is nil", 2) end
	if   ab == nil then	error("Tensor Name is nil", 2) end
	if    x == nil then	error("X Offset is nil", 2) end
	if    y == nil then	error("Y Offset is nil", 2) end

	x = x + 1 --c code decs idxs
	y = y + 1
	local q = mag:tensorArray(dest, src, ab)
	if q == nil then
		error("No tensor found at given layers or bad tensor name", 2)
	end
	return q:get(x,y)
end

local function setMatrix(mag, dest, src, ab, x, y,v)
	mag:makeData() -- if needed

	if dest == nil then	error("Destination Layer is nil", 2) end
	if  src == nil then	error("Source Layer is nil", 2) end
	if   ab == nil then	error("Tensor Name is nil", 2) end
	if    x == nil then	error("X Offset is nil", 2) end
	if    y == nil then	error("Y Offset is nil", 2) end
	
	x = x + 1 --c code decs idxs
	y = y + 1
	local q = mag:tensorArray(dest, src, ab)
	if q == nil then
		error("No tensor found at given layers or bad tensor name", 2)
	end
	mag:setCompileRequired(true)
	return q:set(x,y,v)
end


local function setUnitCell(mag, layer, A, B, C)
	if not isNumber(layer) then
		error("Layer expected as first argument",2)
	end
	if layer < 1 or layer > mag:nz() then
		error("Layer must be in the range [1:"..mag:nz().."]",2)
	end
	if not isTable(A) or not isTable(B) or not isTable(C) then
		error("Tables expected for the A, B, C components (2nd through 4th arguments)",2)
	end
	local t = initializeInternalData(mag, mag:internalData())
	
	t.ABC[layer] = {A,B,C}
	
	mag:setInternalData(t)
end

local function unitCell(mag, layer, A, B, C)
	if not isNumber(layer) then
		error("Layer expected as first argument",2)
	end
	if layer < 1 or layer > mag:nz() then
		error("Layer must be in the range [1:"..mag:nz().."]",2)
	end

	local t = initializeInternalData(mag, mag:internalData())
	
	return t.ABC[layer][1], t.ABC[layer][2], t.ABC[layer][3]
end

local function setGrainSize(mag, layer, X, Y, Z)
	if not isNumber(layer) then
		error("Layer expected as first argument",2)
	end
	if layer < 1 or layer > mag:nz() then
		error("Layer must be in the range [1:"..mag:nz().."]",2)
	end
	if not isNumber(X) or not isNumber(Y) or not isNumber(Z) then
		error("Numbers expected for the X, Y, Z components (2nd through 4th arguments)",2)
	end
	local t = initializeInternalData(mag, mag:internalData())
	
	t.grainSize[layer] = {X, Y, Z}
-- 	mag:setStrength(layer, -4.0*math.pi/(X*Y*Z))
	mag:setInternalData(t)
end

local function grainSize(mag, layer)
	if not isNumber(layer) then
		error("Layer expected as argument",2)
	end
	if layer < 1 or layer > mag:nz() then
		error("Layer must be in the range [1:"..mag:nz().."]",2)
	end
	
	local t = initializeInternalData(mag, mag:internalData())
	local x,y,z = t.grainSize[layer][1], t.grainSize[layer][2], t.grainSize[layer][3]
	return x,y,z,x*y*z
end


local function setTruncation(mag, trunc)
	local t = initializeInternalData(mag, mag:internalData())
	t.truncation = trunc
	mag:setInternalData(t)
end

local function truncation(mag, trunc)
	local t = initializeInternalData(mag, mag:internalData())
	return t.truncation
end


-- Magnetostatics2D File I/O routines
local function write_tensor(f, array, name, d, s)
	f:write(name .. "[" .. d .. "][" .. s .. "] = [[\n")
	local nx, ny = array:nx(), array:ny()
	for y=1,ny do
		local t = {}
		for x=1,nx do
			local v = array:get(x,y)
			if v == 0 then
				table.insert(t, "0")
			else
				table.insert(t, string.format("% 16.15E", v))
			end
		end
		f:write(table.concat(t, ", ") .. "\n")
	end
	f:write("]]\n")	
end

local function get_basename(mag)
	return string.format("MAB_%dx%dx%d", mag:nx(), mag:ny(), mag:nz())
end

local function get_possible_files(mag)
    local possibles = {}
	local basename = get_basename(mag)
	local filenames = os.ls()
	for k,v in pairs(filenames) do
		local a,b,x = string.find(v, basename .. "%.(%d+)%.lua")
		if a then
			table.insert(possibles, {v, tonumber(x)})
		end
	end
	return possibles
end

local function get_new_filename(mag)
	local basename = get_basename(mag)
	local fns = get_possible_files(mag)

	local n = {}
	for k,v in pairs(fns) do
		n[v[2]] = 1
	end

	local next = nil
	for i=0,1000 do
		if next == nil then
			if n[i] == nil then
				next = i
			end
		end
	end

	if next == nil then
		next = 0
	end

	return basename .. "." .. next .. ".lua"
end



local function mag2d_save(mag, filename)
	f = io.open(filename, "w")
	if f == nil then
		error("Failed to open `" .. filename .. "' for writing")
	end
	
	f:write([[
-- Magnetostatics2D Data File
--

-- system size
local nx, ny, nz = ]] .. table.concat({mag:nx(), mag:ny(), mag:nz()}, ", ") .. "\n" ..
[[	
-- internal state
local internal = {}
internal.ABC = {}
internal.grainSize = {}
]])

	local internal = mag:internalData()
	for i=1,mag:nz() do
		f:write("internal.ABC[" .. i .. "] = {" .. 
				table.concat( {
					"{" .. table.concat(internal.ABC[i][1], ", ") .. "}",
					"{" .. table.concat(internal.ABC[i][2], ", ") .. "}",
					"{" .. table.concat(internal.ABC[i][3], ", ") .. "}"
				}, ", ") .. "}\n")
	end
	f:write("\n")
	for i=1,mag:nz() do
		f:write("internal.grainSize[" .. i .. "] = {" .. table.concat(internal.grainSize[i], ", ") .. "}\n")
	end
	f:write("internal.truncation = " .. internal.truncation .. "\n")
	f:write("\n")
	f:write("-- interaction tensors. Format is AB[desitination_layer][source_layer]\n")
	tnames = {"XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"}
	
	local tt = {}
	for d=1,mag:nz() do
		table.insert(tt, "{}")
	end
	tt = "{" .. table.concat(tt, ", ") .. "}"
	for k,ab in pairs(tnames) do
		f:write("local " .. ab .. " = " .. tt .. "\n")
	end
	f:write("\n")
	for k,ab in pairs(tnames) do
		for d=1,mag:nz() do
			for s=1,mag:nz() do
				write_tensor(f, mag:tensorArray(d, s, ab), ab, d, s)
			end
		end
	end
	
	-- add logic to data file to interpret datafile
	f:write([[
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
		table.insert(t, tokenizeNumbers(w))
	end
	
	return t
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

local function parse()
	for i=1,nz do
		for j=1,nz do
			XX[i][j] = parseMatrix(XX[i][j])
			XY[i][j] = parseMatrix(XY[i][j])
			XZ[i][j] = parseMatrix(XZ[i][j])

			YX[i][j] = parseMatrix(YX[i][j])
			YY[i][j] = parseMatrix(YY[i][j])
			YZ[i][j] = parseMatrix(YZ[i][j])

			ZX[i][j] = parseMatrix(ZX[i][j])
			ZY[i][j] = parseMatrix(ZY[i][j])
			ZZ[i][j] = parseMatrix(ZZ[i][j])
		end
	end
end
]])
	-- this function makes the given mag2d object look like this one
	f:write(
string.format([[
local function sameInternals(mag)
    local id = mag:internalData()

    if id == nil then
        return false
    end

    local targetx, targety, targetz = %d, %d, %d
    if mag:nx() ~= targetx or mag:ny() ~= targety or mag:nz() ~= targetz then
        return false
    end 

    for k=1,targetz do
    for i=1,3 do
        if id.grainSize[k][i] ~= internal.grainSize[k][i] then
            return false
        end
        for j=1,3 do
            if id.ABC[k][i][j] ~= internal.ABC[k][i][j] then
                return false
            end
        end
    end
    end

    if id.truncation ~= internal.truncation then
        return false
    end

    return true
end
]], mag:nx(), mag:ny(), mag:nz())

.. [[
return sameInternals, function(mag)
	if mag:nx() ~= nx or mag:ny() ~= ny or mag:nz() ~= nz then
		error("Size Mismatch", 2) --report a size mismatch at the calling location
	end
	mag:setInternalData(internal)
	mag:setNewDataRequired(false) --since we are setting it

	parse()
	
	for i=1,nz do
		for j=1,nz do
			local tXX = mag:tensorArray(i, j, "XX")
			local tXY = mag:tensorArray(i, j, "XY")
			local tXZ = mag:tensorArray(i, j, "XZ")

			local tYX = mag:tensorArray(i, j, "YX")
			local tYY = mag:tensorArray(i, j, "YY")
			local tYZ = mag:tensorArray(i, j, "YZ")
			
			local tZX = mag:tensorArray(i, j, "ZX")
			local tZY = mag:tensorArray(i, j, "ZY")
			local tZZ = mag:tensorArray(i, j, "ZZ")

			for x=1,nx do
				for y=1,ny do
					tXX:set(y,x,XX[i][j][x][y])
					tXY:set(y,x,XY[i][j][x][y])
					tXZ:set(y,x,XZ[i][j][x][y])

					tYX:set(y,x,YX[i][j][x][y])
					tYY:set(y,x,YY[i][j][x][y])
					tYZ:set(y,x,YZ[i][j][x][y])

					tZX:set(y,x,ZX[i][j][x][y])
					tZY:set(y,x,ZY[i][j][x][y])
					tZZ:set(y,x,ZZ[i][j][x][y])
				end
			end
		end
	end
	mag:setCompileRequired(true) --need to Fourier Transform Tensors
end
]])
	f:close()
end

local function mag2d_load(mag, filename)
	local f = io.open(filename, "r")
	if f == nil then
		error("Failed to load `" ..filename .. "'", 2)
	end
	
	local data = f:read("*a")
	
	local sameInternals, loadfunc = assert(loadstring(data))()

	if sameInternals(mag) then
		loadfunc(mag)
		return true
	end
	return false
end










t.getMatrix    = getMatrix
t.setMatrix    = setMatrix
t.setUnitCell  = setUnitCell
t.unitCell     = unitCell
t.setGrainSize = setGrainSize
t.grainSize    = grainSize
t.setTruncation= setTruncation
t.truncation   = truncation
--t.save         = mag2d_save
--t.load         = mag2d_load


local help = MODTAB.help

MODTAB.help = function(x)
--[[
	if x == mag2d_save then
		return
			"Save the tensor and internal data of the " .. MODNAME .. " to a MagLua parsable script",
			"1 String: Filename to save to",
			""
	end
	if x == mag2d_load then
		return
			"Load the " .. MODNAME .. "'s tensors and internal data from a file",
			"1 String: Filename to load from",
			""
	end
--]]
	if x == getMatrix then
		return
			"Get an element of the interaction tensor",
			"2 Integers, 1 String, 2 Integers: destination layer (base 1), source layer (base 1), tensor name: (XX, XY, XZ, YX,...), x and y offset, ",
			"1 Number: Tensor element"
	end
	if x == setUnitCell then
		return
			"Set the unit cell dimensions for one of the layers.",
			"1 Integer, 3 Tables of 3 Numbers: Layer number to set (base 1) and the unit cell vectors of that layer.",
			""
	end
	if x == unitCell then
		return
			"Get the unit cell dimensions for one of the layers.",
			"1 Integer: Layer number to get (base 1).",
			"3 Tables of 3 Numbers: The unit cell vectors of that layer"
	end
	if x == setGrainSize then
		return
			"Set the grain size of the prism for a given layer.",
			"1 Integer, 3 Numbers: Layer number to set (base 1) and the X, Y and Z dimension of the prism at that layer.",
			""
	end
	if x == grainSize then
		return
			"Get the grain size of the prism for a given layer.",
			"1 Integer:  Layer number to get (base 1). ",
			"3 Numbers: The X, Y and Z dimension of the prism at that layer."
	end
	if x == setTruncation then
		return
			"Set the truncation site radius (actually a square) in the tensor generation",
			"1 Integer",
			""
	end
	if x == truncation then
		return
			"Get the truncation site radius (actually a square) in the tensor generation",
			"",
			"1 Integer"
	end	
	
	if x == nil then
		return help()
	end
	return help(x)
end


local function makeData(mag)
	-- first we'll see if the data already exists
	local fns = get_possible_files(mag)

	-- try each shoe for a match
	for k,v in pairs(fns) do
		local f = v[1] 

		if mag2d_load(mag, f) then --we don't need to do work
			return
		end
	end


	local id = initializeInternalData(mag, mag:internalData())
	local ABC = id.ABC
	local grainSize = id.grainSize
	local nx, ny, nz = mag:nx(), mag:ny(), mag:nz()
	
	local NXX,NXY,NXZ = {},{},{}
	local NYX,NYY,NYZ = {},{},{}
	local NZX,NZY,NZZ = {},{},{}
	
	for d=1,nz do
		NXX[d],NXY[d],NXZ[d] = {},{},{}
		NYX[d],NYY[d],NYZ[d] = {},{},{}
		NZX[d],NZY[d],NZZ[d] = {},{},{}
		for s=1,nz do
			NXX[d][s] = mag:tensorArray(d, s, "XX")
			NXY[d][s] = mag:tensorArray(d, s, "XY")
			NXZ[d][s] = mag:tensorArray(d, s, "XZ")

			NYX[d][s] = mag:tensorArray(d, s, "YX")
			NYY[d][s] = mag:tensorArray(d, s, "YY")
			NYZ[d][s] = mag:tensorArray(d, s, "YZ")

			NZX[d][s] = mag:tensorArray(d, s, "ZX")
			NZY[d][s] = mag:tensorArray(d, s, "ZY")
			NZZ[d][s] = mag:tensorArray(d, s, "ZZ")
			
			NXX[d][s]:zero()
			NXY[d][s]:zero()
			NXZ[d][s]:zero()
			
			NYX[d][s]:zero()
			NYY[d][s]:zero()
			NYZ[d][s]:zero()
						
			NZX[d][s]:zero()
			NZY[d][s]:zero()
			NZZ[d][s]:zero()
		end
	end
	
	local cumC = {{0,0,0}} --cumulative C vectors
	
	for i=1,nz do
		cumC[i+1] = {cumC[i][1] + ABC[i][3][1],cumC[i][2] + ABC[i][3][2],cumC[i][3] + ABC[i][3][3]} 
	end
	
	-- assuming all A and B vector are the same for all layers
	local ax, ay, az = ABC[1][1][1], ABC[1][1][2], ABC[1][1][3]
	local bx, by, bz = ABC[1][2][1], ABC[1][2][2], ABC[1][2][3]
	
	for dest = 1,nz do
		local dGrain = grainSize[dest]
		local dcx, dcy, dcz = 0, 0, 0
		for j=1,dest do
			dcx = dcx + ABC[j][3][1]
			dcy = dcy + ABC[j][3][2]
			dcz = dcz + ABC[j][3][3]
		end
		for src = 1,nz do
			local sGrain = grainSize[src]
			local scx, scy, scz = 0, 0, 0
			for j=1,src do
				scx = scx + ABC[j][3][1]
				scy = scy + ABC[j][3][2]
				scz = scz + ABC[j][3][3]
			end
			local cx, cy, cz = scx-dcx, scy-dcy, scz-dcz
			local max = id.truncation
			for X=-max,max do
				for Y=-max,max do
			
					local x = X
					local y = Y
					local rx = (ax * x + bx * y + cx)
					local ry = (ay * x + by * y + cy)
					local rz = (az * x + bz * y + cz)
					
-- 					print(rx,ry,rz)
					
					-- wrap to [0:na-1] bounds
					if x < 0 then
						x = x + (math.floor( -x / nx) + 1)* nx
					end
					if y < 0 then
						y = y + (math.floor( -y / ny) + 1)* ny
					end
					
					x = math.mod(x, nx)
					y = math.mod(y, ny)
					
					local vXX = Magnetostatics2D.NXX(rx,ry,rz, dGrain, sGrain)
					local vXY = Magnetostatics2D.NXY(rx,ry,rz, dGrain, sGrain)
					local vXZ = Magnetostatics2D.NXZ(rx,ry,rz, dGrain, sGrain)
					
					local vYX = Magnetostatics2D.NYX(rx,ry,rz, dGrain, sGrain)
					local vYY = Magnetostatics2D.NYY(rx,ry,rz, dGrain, sGrain)
					local vYZ = Magnetostatics2D.NYZ(rx,ry,rz, dGrain, sGrain)
					
					local vZX = Magnetostatics2D.NZX(rx,ry,rz, dGrain, sGrain)
					local vZY = Magnetostatics2D.NZY(rx,ry,rz, dGrain, sGrain)
					local vZZ = Magnetostatics2D.NZZ(rx,ry,rz, dGrain, sGrain)

					
					-- adding one to indices here because the c++ code decrements them
					NXX[dest][src]:addAt(x+1, y+1, vXX)
					NXY[dest][src]:addAt(x+1, y+1, vXY)
					NXZ[dest][src]:addAt(x+1, y+1, vXZ)
					
					NYX[dest][src]:addAt(x+1, y+1, vYX)
					NYY[dest][src]:addAt(x+1, y+1, vYY)
					NYZ[dest][src]:addAt(x+1, y+1, vYZ)
					
					NZX[dest][src]:addAt(x+1, y+1, vZX)
					NZY[dest][src]:addAt(x+1, y+1, vZY)
					NZZ[dest][src]:addAt(x+1, y+1, vZZ)
				end
			end
		end
	end

-- 	for y=1,ny do
-- 		local r={}
-- 		for x=1,nx do
-- 			table.insert(r, NXY[1][1]:get(x,y))
-- 		end
-- 		print(table.concat(r, ", "))
-- 	end
	
	mag:setCompileRequired(true)

	-- save so we can be lazy later
	local fn = get_new_filename(mag)
	mag2d_save(mag, fn)
end

-- create a function that the C code can call to make the longrange2d operator a magnetostatic2d operator
Magnetostatics2D.internalSetup = function(mag)
	mag:setMakeDataFunction(makeData)
	mag:setInternalData(nil)
end
