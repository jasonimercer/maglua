-- Dipole2D


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
    t.offset = {}
    for i=1,dip:nz() do
	t.offset[i] = {0,0,0}
    end
    t.truncation = 25
    t.truncationX = 25
    t.truncationY = 25
    t.truncationZ = 25
    
    return t
end

local function get_basename(dip)
	return string.format("GAB_%dx%dx%d", dip:nx(), dip:ny(), dip:nz())
end

methods["getMatrix"] = {
    "Get an element of the interaction tensor",
    "4 Integers, 1 String: destination layer (base 1), source layer (base 1), x and y offset, tensor name: XX, XY, XZ, etc",
    "1 Number: Tensor element",
    function(dip, dest, src, x, y, ab)
	local q = dip:tensorArray(dest, src, ab)
	return q:get(x,y)
    end
}

local function isNumber(x) 
    return type(x) == type(1)
end

local function isTable(x)
    return type(x) == type({})
end

methods["setUnitCell"] = {
    "Set the unit cell dimensions for one of the layers.",
    "1 Integer, 3 Tables of 3 Numbers: Layer number to set (base 1) and the unit cell vectors of that layer.",
    "",
    function(dip, layer, A, B, C)
	if not isNumber(layer) then
	    error("Layer expected as first argument")
	end
	if layer < 1 or layer > dip:nz() then
	    error("Layer must be in the range [1:"..dip:nz().."]")
	end

	if B == nil then
	    return dip:setUnitCell(layer, A[1], A[2], A[3])
	end

	if not isTable(A) or not isTable(B) or not isTable(C) then
	    error("Tables expected for the A, B, C components (2nd through 4th arguments)")
	end
	local t = initializeInternalData(dip, dip:internalData())
	
	t.ABC[layer] = {A,B,C}
	
	dip:setInternalData(t)
    end
}

methods["unitCell"] = {
    "Get the unit cell dimensions for one of the layers.",
    "1 Integer: Layer number to get (base 1).",
    "3 Tables of 3 Numbers: The unit cell vectors of that layer",
    function(dip, layer, A, B, C)
	if not isNumber(layer) then
	    error("Layer expected as first argument")
	end
	if layer < 1 or layer > dip:nz() then
	    error("Layer must be in the range [1:"..dip:nz().."]")
	end
	
	local t = initializeInternalData(dip, dip:internalData())
	
	return t.ABC[layer][1], t.ABC[layer][2], t.ABC[layer][3]
    end
}


methods["setOffset"] = {
    "Set the offset from the origin of the local unit cell to the site",
    "1 Integer, 1 Table of 3 Numbers: Layer (z value), offset as a table", 
    "",
    function(dip, layer, v)
	if not isNumber(layer) then
	    error("Layer expected as first argument")
	end
	if layer < 1 or layer > dip:nz() then
	    error("Layer must be in the range [1:"..dip:nz().."]")
	end
	if not isTable(v)  then
	    error("Table expected for 2nd argument")
	end
	local t = initializeInternalData(dip, dip:internalData())
	
	local _v = {}
	for x,y in pairs(v) do
	    _v[x] = y
	end
	t.offset[layer] = _v
	
	dip:setInternalData(t)
    end
}

methods["offset"] = {
    "Get the offset from the origin of the local unit cell to the site on the given layer",
    "1 Integer or nil: Layer (z value).",
    "1 Table of 3 Numbers: Offset of layer or table of all offset for all layers if input is nil.",
    function(dip, layer)
	local t = initializeInternalData(dip, dip:internalData())

	if layer == nil then
	    return t.offset
	end

	if layer < 1 or layer > dip:nz() then
	    error("Layer must be in the range [1:"..dip:nz().."]")
	end
	
	return t.offset[layer]
    end
}


methods["setTruncation"] = {
    "Set the truncation site radii in the tensor generation",
    "Up to 4 Integers: Radial truncation, truncation in X direction, truncation in Y, truncation in Z.",
    "",
    function(dip,trunc, truncX, truncY, truncZ)
        local id = initializeInternalData(dip, dip:internalData())
        id.truncation  = trunc  or id.truncation
        id.truncationX = truncX or id.truncationX
        id.truncationY = truncY or id.truncationY
        id.truncationZ = truncZ or id.truncationZ

        dip:setInternalData(id)
    end
}



methods["truncation"] = {
    "Get the truncation site radii in the tensor generation",
    "",
    "Integers: Radial truncation, truncation in X direction, truncation in Y, truncation in Z",
    function(dip)
	local id = initializeInternalData(dip, dip:internalData())
        return id.truncation, id.truncationX, id.truncationY, id.truncationZ
    end
}


local MODNAME = "Dipole2D"
local MODTAB = _G[MODNAME]

-- inject above into existing metatable for Dipole2D operator
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time
for k,v in pairs(methods) do
    t[k] = v[4]
end

-- backup old help function for fallback
local help = MODTAB.help

-- create new help function for methods above
MODTAB.help = function(x)
                  for k,v in pairs(methods) do
                      if x == v[4] then
                          return v[1], v[2], v[3]
                      end
                  end

                  -- fallback to old help if a case is not handled
                  if x == nil then
                      return help()
                  end
                  return help(x)
              end
















-- many of the next several functions are for automatically saving and loading tensor data
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
	f:flush()
    end
    f:write("]]\n")	
end


local function dip2d_load(dip, filename)
    local f = io.open(filename, "r")
    if f == nil then
	error("Failed to load `" ..filename .. "'", 2)
    end
    
    local data = f:read("*a")
    f:close()
    local sameInternals, loadfunc = assert(loadstring(data))()

    if sameInternals(dip) then
	loadfunc(dip)
	return true
    end
    return false
end


-- we'll look in current directory first, then the common (if it exists)
local function get_possible_files(dip, order)
    local possibles = {}
    local basename = get_basename(dip)
    local available = {}
    
    available["local"] = function()
			     for k,v in pairs(os.ls()) do
				 local a,b,x = string.find(v, basename .. "%.(%d+)%.lua$")
				 if a then
				     table.insert(possibles, {v, tonumber(x)})
				 end
			     end
			 end
    
    available["cache"] = function()
			     if LongRange2D.cacheDirectory then
				 if LongRange2D.cacheDirectory() then
				     local filenames =  os.ls( LongRange2D.cacheDirectory() )
				     for k,v in pairs(filenames) do
					 local a,b,x = string.find(v, basename .. "%.(%d+)%.lua$")
					 if a then
					     table.insert(possibles, {v, tonumber(x)})
					 end
				     end
				 end
			     end
			 end
    
    for k,v in pairs(order) do
	available[v]()
    end
    
    return possibles
end

local function get_new_filename(dip)
    local basename = get_basename(dip)
    local fns = {}
    local cache = nil
    
    if LongRange2D.cacheDirectory then
	if LongRange2D.cacheDirectory() then
	    cache = true
	    fns = get_possible_files(dip, {"cache"})
	end
    end
    
    if cache == nil then -- no cache dir defined, write to local
	fns = get_possible_files(dip, {"local"})
    end
    
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
    
    if cache then
	return  LongRange2D.cacheDirectory() .. basename .. "." .. next .. ".lua"
    else
	return basename .. "." .. next .. ".lua"
    end
end


local function dip2d_save(dip, filename)
    local f = io.open(filename, "w")
    if f == nil then
	error("Failed to open `" .. filename .. "' for writing")
    end
    
    f:write([[
-- Dipole2D Data File
--
MagLua_Version = ]] .. version() .. "\n\n" .. [[
-- system size
local nx, ny, nz = ]] .. table.concat({dip:nx(), dip:ny(), dip:nz()}, ", ") .. "\n" ..
[[	
-- internal state
local internal = {}
internal.ABC = {}
internal.offset = {}
]])

	local internal = dip:internalData()
	for i=1,dip:nz() do
		f:write("internal.ABC[" .. i .. "] = {" .. 
				table.concat( {
					"{" .. table.concat(internal.ABC[i][1], ", ") .. "}",
					"{" .. table.concat(internal.ABC[i][2], ", ") .. "}",
					"{" .. table.concat(internal.ABC[i][3], ", ") .. "}"
				}, ", ") .. "}\n")
	end
	f:write("\n")
	for i=1,dip:nz() do
		f:write("internal.offset[" .. i .. "] = {" .. table.concat(internal.offset[i], ", ") .. "}\n")
	end
	f:write("internal.truncation  = " .. internal.truncation .. "\n")
	f:write("internal.truncationX = " .. internal.truncationX .. "\n")
	f:write("internal.truncationY = " .. internal.truncationY .. "\n")
	f:write("internal.truncationZ = " .. internal.truncationZ .. "\n")
	f:write("\n")
	f:write("-- interaction tensors. Format is AB[destination_layer][source_layer]\n")
	tnames = {"XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"}
	
	local tt = {}
	for d=1,dip:nz() do
		table.insert(tt, "{}")
	end
	tt = "{" .. table.concat(tt, ", ") .. "}"
	for k,ab in pairs(tnames) do
		f:write("local " .. ab .. " = " .. tt .. "\n")
	end
	f:write("\n")
	for k,ab in pairs(tnames) do
		for d=1,dip:nz() do
			for s=1,dip:nz() do
				write_tensor(f, dip:tensorArray(d, s, ab), ab, d, s)
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
			]])
			for k,v in pairs(tnames) do
				f:write( v .. "[i][j] = parseMatrix(" .. v .. "[i][j])\n")
			end
f:write([[
		end
	end
end
]])
	-- this function makes the given dip2d object look like this one
	f:write(
string.format([[
local function sameInternals(dip)
    local id = dip:internalData()
    
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
    if not sameNumber(dip:nx(), targetx) or not sameNumber(dip:ny(), targety) or not sameNumber(dip:nz(), targetz) then
        return false
    end 

    for k=1,targetz do
    for i=1,3 do
        if not sameNumber(id.offset[k][i], internal.offset[k][i]) then
            return false
        end
        for j=1,3 do
            if not sameNumber(id.ABC[k][i][j], internal.ABC[k][i][j]) then
                return false
            end
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
]], dip:nx(), dip:ny(), dip:nz())

.. [[
return sameInternals, function(dip)
	if dip:nx() ~= nx or dip:ny() ~= ny or dip:nz() ~= nz then
		error("Size Mismatch", 2) --report a size mismatch at the calling location
	end
	dip:setInternalData(internal)
	dip:setNewDataRequired(false) --since we are setting it

	parse()
	
	for i=1,nz do
		for j=1,nz do]] .. "\n")
		
	for k,v in pairs(tnames) do
		f:write("local t" .. v .. " = dip:tensorArray(i, j, \"" .. v .. "\")\n")
	end

	f:write([[			for x=1,nx do
				for y=1,ny do]] .. "\n")
	
	for k,v in pairs(tnames) do
		f:write("					t" .. v .. ":set(x,y," .. v .. "[i][j][y][x])\n")
	end
	f:write([[
				end
			end
		end
	end
	dip:setCompileRequired(true) --need to Fourier Transform Tensors
end
]])
	f:close()
end




-- assuming each layer has the same B, C unit cell dimensions
-- ij base 0
-- layer base 1
-- ABC = table(layers) of tables (ABC) of tables (vectors)
-- off = single offset = vector
local function idx2pos(i,j,layer, ABC, off)
    -- dir_z is the sum of the layers up to this layer
    local dir_z = {0,0,0}

    for Z=1,layer-1 do
	local ABCzc = ABC[Z][3]

	dir_z = {dir_z[1] + ABCzc[1], dir_z[2] + ABCzc[2], dir_z[3] + ABCzc[3]}
    end

    local ABC1 = ABC[1]

    local dir_i = {i * ABC1[1][1], i * ABC1[1][2], i * ABC1[1][3]} 
    local dir_j = {j * ABC1[2][2], j * ABC1[2][2], j * ABC1[2][3]} 
    

    return {dir_i[1] + dir_j[1] + dir_z[1] + off[1],
	    dir_i[2] + dir_j[2] + dir_z[2] + off[2],
	    dir_i[3] + dir_j[3] + dir_z[3] + off[3]}

end

local function makeData(dip)
    -- first we'll see if the data already exists
    local fns = get_possible_files(dip, {"local", "cache"})
    
    -- try each shoe for a match
    for k,v in pairs(fns) do
	local f = v[1] 
	if dip2d_load(dip, f) then --we don't need to do work
	    return
	end
    end
    
    local id = initializeInternalData(dip, dip:internalData())
    local ABC = id.ABC
    local offset = id.offset

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


    for dest = 1,nz do
	for src = 1,nz do
	    if math.abs(src - dest) <= id.truncationZ then
		for X=-id.truncationX,id.truncationX do
		    for Y=-id.truncationY,id.truncationY do
			local p_dest = idx2pos(X,Y,dest, ABC, offset[dest])
			local p_src  = idx2pos(X,Y,src , ABC, offset[src ])
			
			local rx = (p_src[1] - p_dest[1])
			local ry = (p_src[2] - p_dest[2])
			local rz = (p_src[3] - p_dest[3])

			local x,y = X,Y
		
			-- wrap to [0:na-1] bounds
			if x < 0 then
			    x = x + 100000*nx
			end
			if y < 0 then
			    y = y + 100000*ny
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
    end	
    dip:setCompileRequired(true)

    -- save so we can be lazy later
    local fn = get_new_filename(dip)
    dip2d_save(dip, fn)
end

-- create a function that the C code can call to make the longrange2d operator a dipnetostatic2d operator
Dipole2D.internalSetup = function(dip)
	dip:setMakeDataFunction(makeData)
	dip:setInternalData(nil)
end









