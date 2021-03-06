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
        for k,v in pairs(arg[1]) do
            cutplanes[k] = v
        end
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

methods["tally"] =
{
"Count the number of elements that fall in given ranges.",
"N Numbers or 1 Table of N Numbers: The division points in the ranges. There is an implicit division point at :min()-1 and :max()+1 at the beginnign and end of the list of numbers.",
"N+1 Integers or 1 Table of N+1 Integers: The count of elements e that satisfy the expression div[i] < e <= div[i+1].",
function(a,...)
    local input_as_table = nil
    if type(arg[1]) == type({}) then
        input_as_table = arg[1]
    else
        input_as_table = arg
    end

    local min = a:min()-1
    local max = a:max()+1
    local res_as_table = a:_tally(input_as_table, min,max)

    if type(arg[1]) == type({}) then
        return res_as_table
    end

    local unwrap_table = nil
    unwrap_table = function(t)
                       if t[1] == nil then
                           return 
                       end
                       local v = t[1]
                       table.remove(t, 1)
                       return v, unwrap_table(t)
                   end
    
    return unwrap_table(res_as_table)
    
end
}

methods["matDiagonal"] =
{
"Extract the values along the diagonal of a matrix",
"1 Array, 1 Optional Array or 1 Optional table: Source matrix and optional destination. If the destination is an array, the diagonal elements will be placed in it if it is of correct dimensions. If it is a table, the diagonal elements will be placed in it. If no destination is supplied a new array will be created and the diagonal elements will be placed along the first row (y = 1).",
"1 Array or 1 Table: The diagonal elements, if no destination is supplied then an Array will be returned.",
function(A,dest)
    if dest == nil then
        dest = A:copy() 
        if dest:ny() > dest:nx() then
            dest = dest:matTrans()
        end
        dest = dest:slice({{1,1}, {dest:nx(),1}})
    end

    local Asz = math.min(A:nx(), A:ny())
    if type(dest) == type({}) then
        for i=1,Asz do
            dest[i] = A:get(i,i)
        end
        return dest
    end

    if type(dest) == type(A) then -- hope it's an array
        local dx, dy = dest:nx(), dest:ny()
        if math.max(dx,dy) < Asz then
            error("Destination size mismatch")
        end

        if dx > dy then
            for i=1,Asz do
                dest:set(i,1, A:get(i,i))
            end
        else
            for i=1,Asz do
                dest:set(1,i, A:get(i,i))
            end
        end
        return dest
    end

    error("Unknown destination type")
end
}

methods["matRank"] =
{
"Determine the rak of a matrix",
"1 Array: Matrix to compute rank",
"1 Integer: Rank of matrix",
function(A)
    local U,S,VT = A:matSVD()

    local dS = S:matDiagonal()

    dS = dS:absoluted(ds)

    local order = dS:get(1,1)

    local zero, finite = dS:tally(order * 1e-20)

    return finite
end
}

methods["matPrint"] =
{
"Print an array as if were a matrix.",
"1 Optional String, 1 Optional Table: If a string is provided, it will be printed to the left of the matrix with matrix dimensions either under the label or to the right of the label if the matrix only has a single row. If a table is provided it will be interpreted as a JSON style options list with keys print (default print), format (default \"% 06.6e\"), delimit (default \"\\t\"), post (default nil) and mathematica (default false).",
"Terminal output: Matrix form",
function(M, name, json)
    if type(name) == type({}) then
        json = name
    end
    json = json or {}
    local format = json.format or "% 06.6e"
    local delim = json.delimit or "\t"
    local p = json.print or print
    local row_labels = json.rowLabels

    local max_row_label_len = 0
    local row_label_fmt = ""
    if row_labels then
        for k,v in pairs(row_labels) do
            local l = string.len(tostring(v))
            if l > max_row_label_len then
                max_row_label_len = l
            end
        end
        row_label_fmt = "%" .. max_row_label_len .. "s"
    end

    local function rl(i, delim) -- row_label
        delim = delim or ""
        if max_row_label_len > 0 then
            return string.format(row_label_fmt, tostring(row_labels[i])) .. delim
        end
    end

    if json.mathematica then
        if name then
            p(name .. " = {")
        else
            p("{")
        end

        local mat = {}
        for r=1,M:ny() do
            local line = {}
            for c=1,M:nx() do
                local s = string.format("% 18.18g", M:get(c,r,1))
                s = string.gsub(s, "e", " 10^")
                line[c] = s
            end
            mat[r] = "  {" .. table.concat(line, ", ") .. "}"
        end

        p(table.concat(mat, ",\n"))
        p("}")
        return
    end

    if name then
	local name_len = string.len(name)
	local dims_txt = M:ny() .. "x" .. M:nx()
	local dims_len = string.len(dims_txt)
	
	if M:ny() == 1 then
	    local t = {}
	    for c=1,M:nx() do
		table.insert(t, string.format(format, M:get(c,1)))
	    end
	    p( name .. "(" .. dims_txt .. ")" .. delim .. (rl(1, delim) or "") .. table.concat(t, delim))
	else
            local l1 = name
            local l2 = "(" .. dims_txt .. ")"

            local max = math.max(string.len(l1), string.len(l2))

            while string.len(l1) < max do
                l1 = " " .. l1 .. " "
            end
            while string.len(l2) < max do
                l2 = " " .. l2 .. " "
            end

            l1 = string.sub(l1, 1, max)
            l2 = string.sub(l2, 1, max)
            local l3 = string.rep(" ", max)

	    local default = {l1, l2, l3}
	    for r=1,M:ny() do
		local t = {default[r] or default[3]}
                local rr = rl(r)
                if rr then
                    table.insert(t, rr)
                end
		for c=1,M:nx() do
		    table.insert(t, string.format(format, M:get(c,r)))
		end
		p(table.concat(t, delim))
	    end
	end
    else
	for r=1,M:ny() do
	    local t = {}
            local rr = rl(r)
            if rr then
                table.insert(t, rr)
            end
	    for c=1,M:nx() do
		table.insert(t, string.format(format, M:get(c,r)))
	    end
	    p(table.concat(t, delim))
	end
    end

    if json.post then
        p(json.post)
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

local function ND_to_flat(t)
    local d1 = table.maxn(t)
    local flat_list = {}

    if type(t[1]) == type({}) then
        local d2 = table.maxn(t[1])
        if type(t[1][1]) == type({}) then
            
            local d3 = table.maxn(t[1][1])
            if type(t[1][1][1]) == type({}) then
                
                for k1=1,d1 do
                    for k2=1,d2 do
                        for k3=1,d3 do
                            table.insert(flat_list, t[k1][k2][k3])
                        end
                    end
                end
                return d3,d2,d1, flat_list
            end
        else
            for k1=1,d1 do
                for k2=1,d2 do
                    table.insert(flat_list, t[k1][k2])
                end
            end
            return d2,d1,1,flat_list
        end
    else
        for k1=1,d1 do
            table.insert(flat_list, t[k1])
        end
        return d1,1,1,flat_list
    end
end

-- augmenting constructors to deal with tables
-- internal constructor will take up to 3 numbers for array size
-- and a flat list with data. This augmented c'tor will accept an array
-- size as a list of numbers. If the sizes are missing
-- they can be inferred from he table of data.
for _,arr in pairs({Array.Double, Array.Float, Array.Integer}) do
    local new = arr.new

    function arr.new(a,b,c,d)
        local n = {a,b,c}
        for i=1,3 do
            if type(n[i]) == type({}) then
                n[i] = 1
            end
            n[i] = n[i] or 1
        end

        if type(a) == type({}) then -- need to infer sizes
            local nx,ny,nz,data =  ND_to_flat(a)
            return new(nx,ny,nz,data)
        else
            return new(a,b,c,d)
        end
    end


end

