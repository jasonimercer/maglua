-- Magnetostatics3D
local MODNAME = "Magnetostatics3D"
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
	t.truncationX = 25
	t.truncationY = 25
	t.truncationZ = 25
	t.ABC       = {{1,0,0}, {0,1,0}, {0,0,1}}
	t.grainSize =  {1,1,1}
	return t
end

local function isNumber(x) 
	return type(x) == "number"
end
local function isTable(x)
	return type(x) == "table"
end

local function getMatrix(mag, ab, x, y, z)
	mag:makeData() -- if needed
	if   ab == nil then	error("Tensor Name is nil", 2)       end
	if    x == nil then	error("X Offset is nil", 2)          end
	if    y == nil then	error("Y Offset is nil", 2)          end

	local q = mag:tensorArray(ab)
	if q == nil then
		error("Bad tensor name", 2)
	end
	return q:get(x+1,y+1,z+1) --c code decs idxs
end

local function setMatrix(mag, ab, x, y, z, v)
	mag:makeData() -- if needed
	if   ab == nil then	error("Tensor Name is nil", 2)       end
	if    x == nil then	error("X Offset is nil", 2)          end
	if    y == nil then	error("Y Offset is nil", 2)          end
	if    z == nil then	error("Z Offset is nil", 2)          end

	local q = mag:tensorArray(ab)
	if q == nil then
		error("Bad tensor name", 2)
	end
	mag:setCompileRequired(true)
	return q:set(x+1,y+1,z+1,v) --c code decs idxs
end


local function setUnitCell(mag, A, B, C)
	if not isTable(A) or not isTable(B) or not isTable(C) then
		error("Tables expected for the A, B, C components (2nd through 4th arguments)",2)
	end
	local id = initializeInternalData(mag, mag:internalData())
	
	id.ABC = {A,B,C}
	mag:setInternalData(id)
end

local function unitCell(mag, A, B, C)
	local id = initializeInternalData(mag, mag:internalData())
	
	return id.ABC[1], id.ABC[2], id.ABC[3]
end

local function setGrainSize(mag, X, Y, Z)
	if not isNumber(X) or not isNumber(Y) or not isNumber(Z) then
		error("Numbers expected for the X, Y, Z components (1st through 3rd arguments)",2)
	end
	local id = initializeInternalData(mag, mag:internalData())
	
	id.grainSize = {X, Y, Z}
	mag:setInternalData(id)
end

local function grainSize(mag)
	local id = initializeInternalData(mag, mag:internalData())
	local x,y,z = id.grainSize[1], id.grainSize[2], id.grainSize[3]
	return x,y,z,x*y*z
end


local function setTruncation(mag, trunc, truncX, truncY, truncZ)
	local id = initializeInternalData(mag, mag:internalData())
	id.truncation  = trunc  or id.truncation
	id.truncationX = truncX or id.truncationX
	id.truncationY = truncY or id.truncationY
	id.truncationZ = truncZ or id.truncationZ

	mag:setInternalData(id)
end

local function truncation(mag)
	local id = initializeInternalData(mag, mag:internalData())
	return id.truncation, id.truncationX, id.truncationY, id.truncationZ
end


-- many of the next several functions are for automatically saving and loading tensor data
local function write_tensor(f, array, name)
	local nx, ny = array:nx(), array:ny()
	f:write(name .. " = [[") -- \ n
	f:write(checkpointToString(array))
	
-- 	for y=1,ny do
-- 		local t = {}
-- 		for x=1,nx do
-- 			local v = array:get(x,y,z)
-- 			if v == 0 then
-- 				table.insert(t, "0")
-- 			else
-- 				table.insert(t, string.format("% 16.15E", v))
-- 			end
-- 		end
-- 		f:write(table.concat(t, ", ") .. "\n")
-- 	end
	f:write("]]\n")	
end

local function get_basename(mag)
	return string.format("M3AB_%dx%dx%d", mag:nx(), mag:ny(), mag:nz())
end

local function get_possible_files(mag)
    local possibles = {}
	local basename = get_basename(mag)
	local filenames = os.ls()
	for k,v in pairs(filenames) do
		local a,b,x = string.find(v, basename .. "%.(%d+)%.lua$")
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
	for i=0,1000 do --looking for a free number
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

local function mag3d_save(mag, filename)
	local f = io.open(filename, "w")
	if f == nil then
		error("Failed to open `" .. filename .. "' for writing")
	end
	
	f:write([[
-- Magnetostatics3D Data File
--
MagLua_Version = ]] .. version() .. "\n\n" .. [[
-- system size
local nx, ny, nz = ]] .. table.concat({mag:nx(), mag:ny(), mag:nz()}, ", ") .. "\n" ..
[[	
-- internal state
local internal = {}
]])

	local internal = mag:internalData()
	f:write("internal.ABC = {" .. 
			table.concat( {
				"{" .. table.concat(internal.ABC[1], ", ") .. "}",
				"{" .. table.concat(internal.ABC[2], ", ") .. "}",
				"{" .. table.concat(internal.ABC[3], ", ") .. "}"
			}, ", ") .. "}\n")

	f:write("\n")
	f:write("internal.grainSize = {" .. table.concat(internal.grainSize, ", ") .. "}\n")

	f:write("internal.truncation  = " .. internal.truncation .. "\n")
	f:write("internal.truncationX = " .. internal.truncationX .. "\n")
	f:write("internal.truncationY = " .. internal.truncationY .. "\n")
	f:write("internal.truncationZ = " .. internal.truncationZ .. "\n")
	f:write("\n")
	tnames = {"XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"}
	
	f:write("local " .. table.concat(tnames, ", ") .. "\n")
	
-- 	for k,ab in pairs(tnames) do
-- 		f:write("local " .. ab .. "\n")
-- 	end
	f:write("\n")
	for k,ab in pairs(tnames) do
-- 		for z=1,mag:nz() do
			write_tensor(f, mag:tensorArray(ab), ab)
-- 		end
	end
	
	
-- 	print("XX tensor:")
-- 	local XX = mag:tensorArray("XX")
-- 	for z=1,XX:nz() do
-- 	for y=1,XX:ny() do
-- 	for x=1,XX:nx() do
-- 		print(x,y,z,XX:get(x,y,z))
-- 	end end end
	
	-- add logic to data file to interpret datafile
-- 	f:write([[
-- local function tokenizeNumbers(line)
-- 	local t = {}
-- 	for w in string.gmatch(line, "[^,]+") do
-- 		table.insert(t, tonumber(w))
-- 	end
-- 	return t
-- end
-- 
-- local function tokenizeLines(lines)
-- 	-- strip empty lines
-- 	lines = string.gsub(lines, "^%s*\n*", "")
-- 	lines = string.gsub(lines, "\n\n+", "\n")
-- 	
-- 	local t = {}
-- 	for w in string.gmatch(lines, "(.-)\n" ) do
-- 		table.insert(t, tokenizeNumbers(w))
-- 	end
-- 	
-- 	return t
-- end
-- 
-- local function parseMatrix(M)
-- 	if M == 0 then
-- 		-- returns a 2D table that always returns zero
-- 		local tz, ttz = {}, {}
-- 		setmetatable(tz,  {__index = function() return  0 end})
-- 		setmetatable(ttz, {__index = function() return tz end})
-- 		return ttz
-- 	end
-- 	
-- 	return tokenizeLines(M)
-- end
-- 
-- local function parse()
-- 	for i=1,nz do
-- 			]])
-- 			for k,v in pairs(tnames) do
-- 				f:write( v .. "[i] = parseMatrix(" .. v .. "[i])\n")
-- 			end
-- f:write([[
-- 	end
-- end
-- ]])
	-- this function makes the given mag3d object look like this one
	f:write(
string.format([[
local function sameInternals(mag)
    local id = mag:internalData()

    if id == nil then
        return false
    end

	local function sameNumber(a,b)
		if a == b then
			return true
		end

		if a == 0 or b == 0 then
			return false
		end

		local c = math.abs(a-b)
		local d = math.abs(a+b)
		if c/d < 1e-8 then
			return true
		end
		return false
	end
	
	local targetx, targety, targetz = %d, %d, %d
    if not 	sameNumber(mag:nx(), targetx) or not sameNumber(mag:ny(), targety) or not sameNumber(mag:nz(), targetz) then
        return false
    end 

    for i=1,3 do
        if not sameNumber(id.grainSize[i], internal.grainSize[i]) then
            return false
        end
        for j=1,3 do
            if not sameNumber(id.ABC[i][j], internal.ABC[i][j]) then
                return false
            end
        end
    end

    if not sameNumber(id.truncation,  internal.truncation)  then
        return false
    end
	if not sameNumber(id.truncationX, internal.truncationX) then
        return false
    end
	if not sameNumber(id.truncationY, internal.truncationY) then
        return false
    end
	if not sameNumber(id.truncationZ, internal.truncationZ) then
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

	]] .. "\n")
		
	for k,v in pairs(tnames) do
		f:write("	mag:setTensorArray(\"" .. v .. "\", checkpointFromString(" .. v .. "))\n")
		f:write("	collectgarbage()\n")
	end

	f:write([[
	mag:setCompileRequired(true) --need to Fourier Transform Tensors
end
]])
	f:close()
end

local function mag3d_load(mag, filename)
	local f = io.open(filename, "r")
	if f == nil then
		error("Failed to load `" ..filename .. "'", 2)
	end
	
	local data = f:read("*a")
	f:close()
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
	if x == getMatrix then
		return
			"Get an element of the interaction tensor",
			"1 String, 3 Integers: tensor name: (XX, XY, XZ, YX,...), x, y and z offset, ",
			"1 Number: Tensor element"
	end
	if x == setUnitCell then
		return
			"Set the unit cell dimensions.",
			"3 Tables of 3 Numbers: Unit cell vectors.",
			""
	end
	if x == unitCell then
		return
			"Get the unit cell dimensions.",
			"",
			"3 Tables of 3 Numbers: The unit cell vectors."
	end
	if x == setGrainSize then
		return
			"Set the grain size of the prism.",
			"3 Numbers: The X, Y and Z dimension of the prism.",
			""
	end
	if x == grainSize then
		return
			"Get the grain size of the prism.",
			"",
			"3 Numbers: The X, Y and Z dimension of the prism."
	end
	if x == setTruncation then
		return
			"Set the truncation site radii in the tensor generation",
			"Up to 4 Integers: Radial truncation, truncation in X direction, truncation in Y, truncation in Z",
			""
	end
	if x == truncation then
		return
			"Get the truncation site radii in the tensor generation",
			"",
			"Integers: Radial truncation, truncation in X direction, truncation in Y, truncation in Z"
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
		if mag3d_load(mag, f) then --we don't need to do work
			return
		end
	end


	local id = initializeInternalData(mag, mag:internalData())
	local ABC = id.ABC
	local grainSize = id.grainSize
	local nx, ny, nz = mag:nx(), mag:ny(), mag:nz()
	
	local NXX,NXY,NXZ = nil,nil,nil
	local NYX,NYY,NYZ = nil,nil,nil
	local NZX,NZY,NZZ = nil,nil,nil
	

	local NXX = mag:tensorArray("XX")
	local NXY = mag:tensorArray("XY")
	local NXZ = mag:tensorArray("XZ")

	local NYX = mag:tensorArray("YX")
	local NYY = mag:tensorArray("YY")
	local NYZ = mag:tensorArray("YZ")

	local NZX = mag:tensorArray("ZX")
	local NZY = mag:tensorArray("ZY")
	local NZZ = mag:tensorArray("ZZ")
			
	NXX:zero()
	NXY:zero()
	NXZ:zero()
	
	NYX:zero()
	NYY:zero()
	NYZ:zero()
			
	NZX:zero()
	NZY:zero()
	NZZ:zero()
	
	-- max function
	local function mf(a,b)
		if a>b then
			return a
		end
		return b
	end
	
	local max = mf(id.truncation, mf(id.truncationX, mf(id.truncationY, id.truncationZ)))

	-- assuming all A and B vector are the same for all layers
	local ax, ay, az = ABC[1][1], ABC[1][2], ABC[1][3]
	local bx, by, bz = ABC[2][1], ABC[2][2], ABC[2][3]
	local cx, cy, cz = ABC[3][1], ABC[3][2], ABC[3][3]
	
-- 	print(ax,ay,az)
-- 	print(bx,by,bz)
-- 	print(cx,cy,cz)
	
	for Z=-max,max do
		if math.abs(Z) <= id.truncationZ then
			for Y=-max,max do
				if math.abs(Y) <= id.truncationY then
					for X=-max,max do
						if math.abs(X) <= id.truncationX then
							local x, y, z = X, Y, Z
							
							local rx = (ax * x + bx * y + cx * z)
							local ry = (ay * x + by * y + cy * z)
							local rz = (az * x + bz * y + cz * z)
					
							-- wrap to [0:na-1] bounds
							while x < 0 do
								x = x + 1000*nx
							end
							while y < 0 do
								y = y + 1000*ny
							end
							while z < 0 do
								z = z + 1000*nz
							end
							
							x = math.mod(x, nx)
							y = math.mod(y, ny)
							z = math.mod(z, nz)
-- 							print(x,y,z,rx,ry,rz)
							local vXX = Magnetostatics2D.NXX(rx,ry,rz, grainSize, grainSize)
							local vXY = Magnetostatics2D.NXY(rx,ry,rz, grainSize, grainSize)
							local vXZ = Magnetostatics2D.NXZ(rx,ry,rz, grainSize, grainSize)
							
							local vYX = Magnetostatics2D.NYX(rx,ry,rz, grainSize, grainSize)
							local vYY = Magnetostatics2D.NYY(rx,ry,rz, grainSize, grainSize)
							local vYZ = Magnetostatics2D.NYZ(rx,ry,rz, grainSize, grainSize)
							
							local vZX = Magnetostatics2D.NZX(rx,ry,rz, grainSize, grainSize)
							local vZY = Magnetostatics2D.NZY(rx,ry,rz, grainSize, grainSize)
							local vZZ = Magnetostatics2D.NZZ(rx,ry,rz, grainSize, grainSize)
							
							-- adding one to indices here because the c++ code decrements them
							NXX:addAt(x+1, y+1, z+1, vXX)
							NXY:addAt(x+1, y+1, z+1, vXY)
							NXZ:addAt(x+1, y+1, z+1, vXZ)
							
							NYX:addAt(x+1, y+1, z+1, vYX)
							NYY:addAt(x+1, y+1, z+1, vYY)
							NYZ:addAt(x+1, y+1, z+1, vYZ)
							
							NZX:addAt(x+1, y+1, z+1, vZX)
							NZY:addAt(x+1, y+1, z+1, vZY)
							NZZ:addAt(x+1, y+1, z+1, vZZ)						
						end
					end
				end
			end
		end
	end

	mag:setCompileRequired(true)

	-- save so we can be lazy later
	local fn = get_new_filename(mag)
	mag3d_save(mag, fn)
end

-- create a function that the C code can call to make the longrange3d operator a magnetostatic3d operator
Magnetostatics3D.internalSetup = function(mag)
	mag:setMakeDataFunction(makeData)
	-- the following used to be not commented out but gives problems when checkpoint loading
	-- mag:setInternalData(nil) 
end
