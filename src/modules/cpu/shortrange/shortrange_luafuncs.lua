-- ShortRange
local MODNAME = "ShortRange"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time


local methods = {}

methods["add"] =
{
"Create a connection between two sites where one generates a field and the other recieves the field.",
"2 Tables of Integers, 1 Number, 1 Matrix, 1 Optional Number: The two tables of integers represent locations (as field source and field destination), missing table elements will be assumed to be 1. The Number is a scaling factor, the Matrix can be a table of 9 numbers or a table of tables of 3 numbers or an Array.Double and define how different Cartesian linearly combine to create the field. The last optional number is the exponent of the source site configuration dotted with itself. A zero is the default and will make this term 1.",
"",
function(sr, src, dest, strength, gab, e)
    if type(src) ~= type({}) then
	error("Expected table for first argument")
    end
    if type(dest) ~= type({}) then
	error("Expected table for second argument")
    end
    
    for i=1,3 do
	src[i] = src[i] or 1
	dest[i] = dest[i] or 1
    end

    if type(strength) ~= type(1) then
	error("Expeted number of third arguement")
    end

    local _gab = {}
    -- working on matrix:
    if type(gab) == type({}) then --it's a table
	if type(gab[1]) == type({}) then -- table of tables, let's flatten it
	    for i=1,3 do
		for j=1,3 do
		    table.insert(_gab, gab[i][j] or 0)
		end
	    end
	else -- assuming flat table
	    for i=1,9 do
		_gab[i] = gab[i] or 0
	    end
	end
    else
	if getmetatable(gab) == Array.Double.metatable() then -- it's an Array.Double
	    for i=1,3 do
		for j=1,3 do
		    table.insert(_gab, gab.get(i,j) or 0)
		end
	    end
	else
	    error("Unknown datatype for matrix")
	end
    end

    e = e or 0

    return sr:_addPath(src, dest, strength, _gab, e)
end
}

local f = methods["add"][4]
methods["addPath"] = 
{
"Synonym for add.",
"Same input as add.",
"",
function(...)
    return f(...)
end
}


methods["addMagnetostatic2D"] =
    {
    "Add a field rule from a Magnetostatic2D operator. The rule can either use the values in the tensor arrays in the given operator (reproducing the calculated lattice periodicity) or calculate a single term. The unit cell information is read from the Magnetostatic2D operator. The following example would calculate all the Magnetostatic fields acting between the pair of sites {8,8,1} and {8,8,2}:" ..
[[<pre>
sr:addMagnetostatic2D(mag2d, {8,8,1}, {8,8,1}, "Calculate")
sr:addMagnetostatic2D(mag2d, {8,8,1}, {8,8,2}, "Calculate")
sr:addMagnetostatic2D(mag2d, {8,8,2}, {8,8,1}, "Calculate")
sr:addMagnetostatic2D(mag2d, {8,8,2}, {8,8,2}, "Calculate")
</pre>
]],
    [[1 Magnetostatics2D operator, 2 Tables of Integers, 1 String, 1 Optional Number: The magnetostatic operator contains the unit cell information, the tables specify field source and field destination locations, the string must be either "Read" or "Calculate", specifying "Read" will tell this method to read the values from the tensors in the magnetostatic operator, "Calculate" will tell the method to calculate the term (which will not include any lattice periodicity included in the operator). The optional number will artificially scale the read or calculated values. Without this number there is no artificial scaling.]],
    "",
    function(sr, mag, _src, _dest, kind, scale)
	if getmetatable(mag) ~= Magnetostatics2D.metatable() then
	    error("First argument must be a Magnetostatics2D operator")
	end
	if type(_src) ~= type({}) or type(_dest) ~= type({}) then
	    error("Source and Destination must be tables")
	end
	local src, dest = {}, {}
	for i=1,3 do
	    src[i] = _src[i] or 1
	    dest[i] = _dest[i] or 1
	end
	scale = scale or 1
	local nx,ny = mag:nx(), mag:ny()
	local ab = {"XX","XY","XZ","YX","YY","YZ","ZX","ZY","ZZ"}
	local values = {}

	if string.lower(kind) == "read" then
	    -- tensors are offsets. Need to get the offsets from the input
	    local dx = dest[1] - src[1]
	    local dy = dest[2] - src[2]

	    -- wrap to [0:na-1] bounds
	    while dx < 0 do
		dx = dx + 1000*nx
	    end
	    while dy < 0 do
		dy = dy + 1000*ny
	    end
	    
	    dx = math.mod(dx, nx)
	    dy = math.mod(dy, ny)
	    mag:makeData() -- if required
	    for i=1,9 do
		local a = mag:tensorArray(dest[3], src[3], ab[i])
		local v = a:get(dx+1, dy+1)
		values[ ab[i] ] = v
	    end
	end

	-- adding undocumented "compute" because people will expect it to work
	if string.lower(kind) == "calculate" or string.lower(kind) == "compute" then 
	    local r1 = mag:indexToPosition(src)
	    local r2 = mag:indexToPosition(dest)

	    local x,y,z = mag:grainSize(src[3])
	    local sGrain = {x,y,z}

	    local x,y,z = mag:grainSize(dest[3])
	    local dGrain = {x,y,z}

	    -- dest - src
	    local rx,ry,rz = r2[1]-r1[1], r2[2]-r1[2], r2[3]-r1[3]

	    -- negating rz. Need to think about the implications of this
	    -- doing it to match the results of the longrange tensors
	    rz = -rz

	    values["XX"] = Magnetostatics2D.NXX(rx,ry,rz, sGrain, dGrain)
	    values["XY"] = Magnetostatics2D.NXY(rx,ry,rz, sGrain, dGrain)
	    values["XZ"] = Magnetostatics2D.NXZ(rx,ry,rz, sGrain, dGrain)

	    values["YX"] = Magnetostatics2D.NYX(rx,ry,rz, sGrain, dGrain)
	    values["YY"] = Magnetostatics2D.NYY(rx,ry,rz, sGrain, dGrain)
	    values["YZ"] = Magnetostatics2D.NYZ(rx,ry,rz, sGrain, dGrain)

	    values["ZX"] = Magnetostatics2D.NZX(rx,ry,rz, sGrain, dGrain)
	    values["ZY"] = Magnetostatics2D.NZY(rx,ry,rz, sGrain, dGrain)
	    values["ZZ"] = Magnetostatics2D.NZZ(rx,ry,rz, sGrain, dGrain)
	end

	if values["XX"] == nil then
	    error("Unknown type, expected \"Read\" or \"Calculate\"")
	end

	local t = {}
	t[1] = values["XX"]
	t[2] = values["YX"]
	t[3] = values["ZX"]

	t[4] = values["XY"]
	t[5] = values["YY"]
	t[6] = values["ZY"]

	t[7] = values["XZ"]
	t[8] = values["YZ"]
	t[9] = values["ZZ"]

	for i=1,9 do
	    t[i] = t[i] * scale
	end

	-- ok! we now have our tensor elements. Time to add the rule to the sr operator
	local strength = mag:strength(dest[3])
	sr:add(src, dest, strength, t, 0) -- the zero gets rid of the s dot s term, ie (s.s)^0
	return values
    end
}


-- inject above into existing metatable for MEP operator
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
                  if x == nil then -- overview
                      return 
[[Generate fields from one site onto another using a linear combination of source site Cartsian Components, powers of the source site Cartesian components dotted with itself and a scale factor. The field from source site i to destination site j is written as:
	$ \vec{H}_{j} = \left( \begin{smallmatrix} xx&yx&zx\\xy&yy&zy\\xz&yz&zz \end{smallmatrix} \right) \left( \begin{smallmatrix} m_i^x\\m_i^y\\m_i^z \end{smallmatrix} \right) \left( \vec{m}_i \cdot \vec{m}_i \right)^{p} f $.

In MagLua, given a ShortRange operator named sr, this could be added to the operator with the command:
<pre>
i = {1,2,1} -- source
j = {2,2,1} -- destination
sr:add(i, j, f, {{xx,yx,zx},{yx,yy,yz},{zx,zy,zz}}, p)
</pre>
A single ShortRange operator can contain many rules and could reproduce exactly the fields of the Exchange operator and the Long Range operators. The reason for the Short Range operator is to be able to calculate parts of the long range fields without running the full calculation.  
]], "", ""
                  end

		  return help(x)
	      end