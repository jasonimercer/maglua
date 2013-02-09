-- Dipole2D

local MODNAME = "Dipole2D"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

-- at first the internal data is nil. If it is it
-- needs to be setup for dip2d
local function initializeInternalData(dip, t)
	if t ~= nil then
		return t --it's OK
	end
	t = {}
	t.ABC = {}
	for i=1,dip:nz() do
		t.ABC[i]       = {{1,0,0}, {0,1,0}, {0,0,1}}
	end
	return t
end
	
local function getMatrix(dip, dest, src, x, y, ab)
	local q = dip:tensorArray(dest, src, ab)
	return q:get(x,y)
end

local function isNumber(x) 
	return type(x) == "number"
end

local function isTable(x)
	return type(x) == "table"
end

local function setUnitCell(dip, layer, A, B, C)
	if not isNumber(layer) then
		error("Layer expected as first argument")
	end
	if layer < 1 or layer > dip:nz() then
		error("Layer must be in the range [1:"..dip:nz().."]")
	end
	if not isTable(A) or not isTable(B) or not isTable(C) then
		error("Tables expected for the A, B, C components (2nd through 4th arguments)")
	end
	local t = initializeInternalData(dip, dip:internalData())
	
	t.ABC[layer] = {A,B,C}
	
	dip:setInternalData(t)
end

local function unitCell(dip, layer, A, B, C)
	if not isNumber(layer) then
		error("Layer expected as first argument")
	end
	if layer < 1 or layer > dip:nz() then
		error("Layer must be in the range [1:"..dip:nz().."]")
	end

	local t = initializeInternalData(dip, dip:internalData())
	
	return t.ABC[layer][1], t.ABC[layer][2], t.ABC[layer][3]
end

t.getMatrix    = getMatrix
t.setUnitCell  = setUnitCell
t.unitCell     = unitCell


local help = MODTAB.help

MODTAB.help = function(x)
	if x == getMatrix then
		return
			"Get an element of the interaction tensor",
			"4 Integers, 1 String: destination layer (base 1), source layer (base 1), x and y offset, tensor name: XX, XY, XZ, etc",
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
	
	if x == nil then
		return help()
	end
	return help(x)
end


local function makeData(dip)
	local id = initializeInternalData(dip, dip:internalData())
	local ABC = id.ABC
	local nx, ny, nz = dip:nx(), dip:ny(), dip:nz()
	
	local NXX,NXY,NXZ = {},{},{}
	local NYX,NYY,NYZ = {},{},{}
	local NZX,NZY,NZZ = {},{},{}
	
	for d=1,nz do
		NXX[d],NXY[d],NXZ[d] = {},{},{}
		NYX[d],NYY[d],NYZ[d] = {},{},{}
		NZX[d],NZY[d],NZZ[d] = {},{},{}
		for s=1,nz do
			NXX[d][s] = dip:tensorArray(d, s, "XX")
			NXY[d][s] = dip:tensorArray(d, s, "XY")
			NXZ[d][s] = dip:tensorArray(d, s, "XZ")

			NYX[d][s] = dip:tensorArray(d, s, "YX")
			NYY[d][s] = dip:tensorArray(d, s, "YY")
			NYZ[d][s] = dip:tensorArray(d, s, "YZ")

			NZX[d][s] = dip:tensorArray(d, s, "ZX")
			NZY[d][s] = dip:tensorArray(d, s, "ZY")
			NZZ[d][s] = dip:tensorArray(d, s, "ZZ")
			
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
		local dcx, dcy, dcz = 0, 0, 0
		for j=1,dest do
			dcx = dcx + ABC[j][3][1]
			dcy = dcy + ABC[j][3][2]
			dcz = dcz + ABC[j][3][3]
		end
		for src = 1,nz do
			local scx, scy, scz = 0, 0, 0
			for j=1,src do
				scx = scx + ABC[j][3][1]
				scy = scy + ABC[j][3][2]
				scz = scz + ABC[j][3][3]
			end
			local cx, cy, cz = scx-dcx, scy-dcy, scz-dcz
			
			for X=-20,20 do
				for Y=-20,20 do
			
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
					
					local vXX = Dipole2D.NXX(rx,ry,rz)
					local vXY = Dipole2D.NXY(rx,ry,rz)
					local vXZ = Dipole2D.NXZ(rx,ry,rz)
					
					local vYX = Dipole2D.NYX(rx,ry,rz)
					local vYY = Dipole2D.NYY(rx,ry,rz)
					local vYZ = Dipole2D.NYZ(rx,ry,rz)
					
					local vZX = Dipole2D.NZX(rx,ry,rz)
					local vZY = Dipole2D.NZY(rx,ry,rz)
					local vZZ = Dipole2D.NZZ(rx,ry,rz)

					
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
	
	dip:setCompileRequired(true)
end

-- create a function that the C code can call to make the longrange2d operator a dipnetostatic2d operator
Dipole2D.internalSetup = function(dip)
	dip:setMakeDataFunction(makeData)
	dip:setInternalData(nil)
end
