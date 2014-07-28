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




-- adding in some new methods via lua
methods = {}

local function getCutsAndDest(a, arg, dir)
	local cutplanes = {}
	if type(arg[1]) == type({}) then
		-- we have a table of planes to cut
		cutplanes = arg[1]
	else
		-- else we have a list of cuts
		for i=1,table.maxn(arg) do
			if type(arg[i]) == type(1) then
				table.insert(cutplanes, arg[i])
			end
		end
	end
	
	local dest = nil
	for i=1,table.maxn(arg) do
		if type(arg[i]) == type(a) then
			-- we hope it's an array
			dest = arg[i]
		end
	end

	table.sort(cutplanes)
	local num_cuts = table.maxn(cutplanes)

	local tx, ty, tz -- target size 
	
	local x,y,z = a:nx(), a:ny(), a:nz()
	if dir == "x" then 	tx, ty, tz = x - num_cuts, y, z  end
	if dir == "y" then 	tx, ty, tz = x, y - num_cuts, z  end
	if dir == "z" then 	tx, ty, tz = x, y, z - num_cuts  end
	
	if dest == nil then
		dest = a:slice({{1,1,1}, {tx,ty,tz}})
	end

	if dest:nx() ~= tx or dest:ny() ~= ty or dest:nz() ~= tz then
		error("Destination array is wrong dimension", 3)
	end

	return cutplanes, num_cuts, dest
end

methods["cutX"] =
{
    "Get a copy of the array with elements in given X planes removed.",
    "0 or more Integers or a table of Integers, 1 Optional Array: Indices to be removed (In a 2D matrix these would be columns). If an array is provided and is the correct size, it will receive the copy.",
    "1 Array: Result of cut",
    function(a, ...)
		local x,y,z = a:nx(), a:ny(), a:nz()
		local cutplanes, num_cuts, dest = getCutsAndDest(a, arg, "x")

		if num_cuts == 0 then
			a:slice({{1,1,1}, {x,y,z}}, dest)
			return dest
		end

		table.insert(cutplanes, x+1) -- add cut after end

		local s1,s2,d1,d2

		s2, d2 = -1, 0
		for i=1,num_cuts+1 do
			s1, s2 = s2+2, cutplanes[i]-1
			d1, d2 = d2+1, d2+1 + (s2-s1)
			
			if s1 <= s2 then
				a:slice({{s1,1,1}, {s2,y,z}}, dest, {{d1,1,1}, {d2,y,z}})
			end
		end
		return dest
    end
}


methods["cutY"] =
{
    "Get a copy of the array with elements in given Y planes removed.",
    "0 or more Integers or a table of Integers, 1 Optional Array: Indices to be removed (In a 2D matrix these would be rows). If an array is provided and is the correct size, it will receive the copy.",
    "1 Array: Result of cut",
    function(a, ...)
		local x,y,z = a:nx(), a:ny(), a:nz()
		local cutplanes, num_cuts, dest = getCutsAndDest(a, arg, "y")

		if num_cuts == 0 then
			a:slice({{1,1,1}, {x,y,z}}, dest)
			return dest
		end

		table.insert(cutplanes, y+1) -- add cut after end

		local s1,s2,d1,d2

		s2, d2 = -1, 0
		for i=1,num_cuts+1 do
			s1, s2 = s2+2, cutplanes[i]-1
			d1, d2 = d2+1, d2+1 + (s2-s1)
			
			if s1 <= s2 then
				a:slice({{1,s1,1}, {x,s2,z}}, dest, {{1,d1,1}, {x,d2,z}})
			end
		end
		return dest
    end
}

methods["cutZ"] =
{
    "Get a copy of the array with elements in given Z planes removed.",
    "0 or more Integers or a table of Integers, 1 Optional Array: Indices to be removed. If an array is provided and is the correct size, it will receive the copy.",
    "1 Array: Result of cut",
    function(a, ...)
		local x,y,z = a:nx(), a:ny(), a:nz()
		local cutplanes, num_cuts, dest = getCutsAndDest(a, arg, "z")

		if num_cuts == 0 then
			a:slice({{1,1,1}, {x,y,z}}, dest)
			return dest
		end

		table.insert(cutplanes, z+1) -- add cut after end

		local s1,s2,d1,d2

		s2, d2 = -1, 0
		for i=1,num_cuts+1 do
			s1, s2 = s2+2, cutplanes[i]-1
			d1, d2 = d2+1, d2+1 + (s2-s1)
			
			if s1 <= s2 then
				a:slice({{1,1,s1}, {x,y,s2}}, dest, {{1,1,d1}, {x,y,d2}})
			end
		end
		return dest
    end
}


methods["swappedX"] =
{
	"Return an array equal to the calling array with 2 X planes swapped",
	"2 Integers, 1 Optional Array: indices of planes to swap. Optional destination array.",
	"1 Array: Array with planes swapped",
	function(a, x1, x2, d)
		d = a:copy(d)

		local plane1 = d:slice({{x1, 1,1}, {x1, d:ny(), d:nz()}})
		
		d:slice({{x2, 1,1}, {x2, d:ny(), d:nz()}}, d, {{x1, 1,1}, {x1, d:ny(), d:nz()}})
		plane1:slice({{1,1,1}, {1,d:ny(), d:nz()}}, d, {{x2, 1,1}, {x2, d:ny(), d:nz()}})

		return d
	end
}
methods["swappedY"] =
{
	"Return an array equal to the calling array with 2 Y planes swapped",
	"2 Integers, 1 Optional Array: indices of planes to swap. Optional destination array.",
	"1 Array: Array with planes swapped",
	function(a, y1, y2, d)
		d = a:copy(d)

		local plane1 = d:slice({{1,y1,1}, {d:nx(), y1, d:nz()}})
		
		d:slice({{1,y2,1}, {d:nx(), y2, d:nz()}}, d, {{1,y1,1}, {d:nx(), y1, d:nz()}})
		plane1:slice({{1,1,1}, {d:nx(), 1, d:nz()}}, d, {{1,y2,1}, {d:nx(), y2, d:nz()}})

		return d
	end
}
methods["swappedZ"] =
{
	"Return an array equal to the calling array with 2 Z planes swapped",
	"2 Integers, 1 Optional Array: indices of planes to swap. Optional destination array.",
	"1 Array: Array with planes swapped",
	function(a, z1, z2, d)
		d = a:copy(d)

		local plane1 = d:slice({{1,1,z1}, {d:nx(), d:ny(),z1}})
		
		d:slice({{1,1,z2}, {d:nx(), d:ny(),z2}}, d, {{1,1,z1}, {d:nx(), d:ny(), z1}})
		plane1:slice({{1,1,1}, {d:nx(), d:ny(),1}}, d, {{1,1,z2}, {d:nx(), d:ny(), z2}})

		return d
	end
}


methods["matPrint"] =
{
"Print an array as if were a matrix.",
"1 Optional String: If a string is provided, it will be printed to the left of the matrix with matrix dimensions either under the label or to the right of the label if the matrix only has a single row.",
"Terminal output: Matrix form",
function(M, name)
    if name then
	local name_len = string.len(name)
	local dims_txt = M:ny() .. "x" .. M:nx()
	local dims_len = string.len(dims_txt)
	
	if M:ny() == 1 then
	    local t = {}
	    for c=1,M:nx() do
		table.insert(t, string.format("% 06.6e", M:get(c,1)))
	    end
	    print( name .. "(" .. dims_txt .. ")", table.concat(t, "\t"))
	else
	    local default = {"  " .. name, "(" .. dims_txt .. ")"}
	    for r=1,M:ny() do
		local t = {default[r] or ""}
		for c=1,M:nx() do
		    table.insert(t, string.format("% 06.6e", M:get(c,r)))
		end
		print(table.concat(t, "\t"))
	    end
	end
    else
	for r=1,M:ny() do
	    local t = {}
	    for c=1,M:nx() do
		table.insert(t, string.format("% 06.6e", M:get(c,r)))
	    end
	    print(table.concat(t, "\t"))
	end
    end
end
}


for _,name in pairs({"Array.Double", "Array.Float", "Array.DoubleComplex", "Array.Integer"}) do
	local t = maglua_getmetatable(name)

	-- inject above into existing metatable
	for k,v in pairs(methods) do
		t[k] = v[4]
	end

	function get_module()
		return assert(loadstring("return " .. name))()
	end

	-- backup old help function for fallback
	local help = get_module()["help"]

	-- create new help function for methods above
	get_module()["help"] = function(x)
		for k,v in pairs(methods) do
			if x == v[4] then
				return v[1], v[2], v[3]
			end
		end
		
		return help(x)
	end
end

