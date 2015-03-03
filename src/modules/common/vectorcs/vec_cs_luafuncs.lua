-- VecCS
local MODNAME = "VectorCS"
local MODTAB = _G[MODNAME]
local mt = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time
local methods = {}
local functions = {}

methods["toTable"] =
{
"Get Vector as a table of Lua types",
"",
"Table of 3 Numbers and 1 String: Coordinates and Coordinate System Name",
function(v)
    local v1,v2,v3,cs = v:toList()
    return {v1,v2,v3,cs}
end
}

methods["fromTable"] =
{
"Set Vector from a table of Lua types",
"Table of 3 Numbers and 1 Optional String: Coordinates and Coordinate System Name (default current)",
"",
function(v,t)
    local _,_,_,cs = v:toList()
    v:fromList(t[1], t[2], t[3], t[4] or cs)
end
}

methods["__tostring"] =
{
"","","",
    function(v)
        local v1,v2,v3,cs = v:toList()
        return string.format([[{%g, %g, %g, "%s"}]], v1,v2,v3,cs)
    end
}

methods["currentSystem"] =
{
    "Get the name of the current coordinate system",
    "",
    "1 String: Current coordinate system",
    function(vec)
        local _,_,_,cs = vec:toList()
        return cs
    end
}

functions["coordinateSystemHasComponent"] =
{
"Query if the given coordinate system has the given component name.",
"2 Strings: coordinate system name and component name.",
"1 Boolean, 1 Integer: query result and index of component if true.",
    function(cs_name,coord_name)
        local data = functions["coordinateSystemData"][4]()

        local t = nil
        for k,v in pairs(data) do
            if string.lower(k) == string.lower(cs_name) then
                t = v
            end
        end

        if t == nil then
            return false
        end

        for k,v in pairs( t[1] ) do
            if string.lower(v) == string.lower(coord_name) then
                return true, k
            end
        end
        return false
    end

}

methods["hasComponent"] =
{
"Query if the current coordinate system has the given component name.",
"1 String: name.",
"1 Boolean, 1 Integer: query result and index of component if true.",
    function(vec,name)
        local cs = vec:currentSystem()

        return functions["coordinateSystemHasComponent"][4](cs, name)
    end
}

methods["get"] =
{
"Get the value of a component, possibly casting the vector to another coordinate system. In cases where the current system does not have a component with the given name and the desired coordinate system is ambiguous, the code will try to cast Cartesian and then Spherical and then Canonical.",
"1 String: Name of component",
"1 Number: value",
function(vec,name)
    name = string.lower(name)

    local b,i = vec:hasComponent(name)

    if b then
        local t = vec:toTable()
        return t[i]
    end

    local cs = {"Cartesian", "Spherical", "Canonical"}
    for j=1,3 do
        local b,i = functions["coordinateSystemHasComponent"][4](cs[j], name)
        if b then
            vec:convertTo(cs[j])
            local t = vec:toTable()
            return t[i]
        end
    end

    return error("Failed to determine coordinate system or component")

end
}

methods["set"] =
{
"Set a component to a value, possibly casting the vector to another coordinate system. In cases where the current system does not have a component with the given name and the desired coordinate system is ambiguous, the code will try to cast Cartesian and then Spherical and then Canonical.",
"1 String, 1 Number: Name of component, value of component","",
function(vec,name,v)
    name = string.lower(name)

    local b,i = vec:hasComponent(name)

    if b then
        local t = vec:toTable()
        t[i] = v
        vec:fromTable(t)
        return
    end

    local cs = {"Cartesian", "Spherical", "Canonical"}
    for i=1,3 do
        local b,j = functions["coordinateSystemHasComponent"][4](cs[i], name)
        
        if b then
            vec:convertTo(cs[i])
            local t = vec:toTable()
            t[j] = v
            vec:fromTable(t)
            return
        end
    end

    return error(string.format(
                     "Failed to determine coordinate system or component. Name = `%s'.", name or "nil"))
end
}


functions["coordinateSystems"] =
{
"Get a table of available coordinate systems",
"",
"1 Table of Strings: Available coordinate systems",
function()
    return {
        "Cartesian",
        "Spherical",
        "Canonical",
        "SphericalX",
        "SphericalY",
        "CanonicalX",
        "CanonicalY"}
end
}

functions["coordinateSystemData"] =
{
"Get the data describing all coordinate systems",
"",
"1 Table of pairs: Each pair is indexed by keys of coordinate system names. The pairs are tables of the dimension names and dimension desription",
function()
    local data = {}
    data["Cartesian"] = {{"x", "y", "z"}, {"X Axis", "Y Axis", "Z Axis"}}
    data["Spherical"] = {{"r", "p", "t"}, {"Radial", "Azimuthal", "Polar"}}
    data["Canonical"] = {{"r", "phi", "p"},{"Radial", "Azimuthal", "cos(Polar)"}}
    return data
end
}

functions["componentNames"] =
    {
    "Get the short and long names for each coordinate component given a coordinate type",
    "1 String: \"Cartesian\", \"Canonical\" or \"Spherical\".",
    "2 Tables of Strings: Short and long forms",
    function(cs)
        if cs == nil then
            error("Coordinate system required")
        end

        local data = funtions["coordinateSystemData"][4]()
        if cs == nil then
            error("method requires a coordinate system name")
        end
        local t = nil
        for k,v in pairs(data) do
            if string.lower(cs) == string.lower(k) then
                t = v
            end
        end
        if t == nil then
            error("method requires a valid coordinate system name, given = `" .. cs .. "'")
        end

        return t[1], t[2]
    end
}





-- inject above into existing metatable for VecCS object
for k,v in pairs(methods) do
    mt[k] = v[4]
end

-- inject functions
for k,v in pairs(functions) do
    MODTAB[k] = v[4]
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

		  for k,v in pairs(functions) do
		      if x == v[4] then
			  return v[1], v[2], v[3]
		      end
		  end

		  -- fallback to old help if a case is not handled
		  if x == nil then -- VecCS overview
		      return [[Ojbect that represents a point in a coordinate system.]], "Constructor can take a table of 3 numbers and 1 string, a list of 3 numbers and 1 string, another VectorCS or nothing. The 3 numbers are the coordinates and the string names the coordinate system.", ""
		  end
		  return help(x)
	      end


