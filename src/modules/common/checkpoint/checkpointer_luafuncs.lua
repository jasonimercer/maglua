-- Checkpointer
local MODNAME = "Checkpointer"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

-- trying something new for style/readability. 
-- Putting docs and code in a table which will later be used
-- to build metatable and help system.
local methods = {}

local op_names = {"Pack", "Compress", "Base64"}
op_names[0] = "None"
op_names[-1] = "Unknown"


methods["decodeHeader"] = 
{
	"Get information about the header of the encoded data",
	"",
	"3 Strings, 2 Numbers: 3 Character signature, New transform name, Old transform name, new size, old size",
	function(cp)
		local sig, ne, oe, ns, os = cp:_decodeHeader()
		return sig, op_names[ne], op_names[oe], ns, os
	end

}

methods["transformation"] =
{
	"Find the most recent transformation applied to the data",
	"",
	"1 String: Most recent transformation name",
	function(cp)
		local i = cp:_transformation()
		return op_names[i] or "ERROR"
	end
}

methods["transform"] =
{
	"Apply a transformation to the internal data, may be one of: " .. table.concat(op_names, ", "),
	"1 String. Apply the given operation name, case does not matter.",
	"",
	function(cp, op_name)
		op_name = tostring(op_name)
		local k = 0
		for i=1,table.maxn(op_names) do
			if string.lower(op_names[i]) == string.lower(op_name) then
				k = i
			end
		end
		if k == 0 then
			error("Failed to match `" .. op_name .. "' to a valid operation")
		end
		cp:_operate(k)
	end
}


methods["untransform"] =
{
	"Apply the inverse of the transformation used to generated the current internal state. ",
	"",
	"",
	function(cp)
		cp:_deoperate()
	end
}

methods["transformations"] = 
{
	"Get a list of operations used to transform the internal state",
	"",
	"1 Table of strings: Operations used to transform the internal state",
	function(cp)
		local t = cp:_operations()
		local r = {}
		for k,v in pairs(t) do
			r[k] = op_names[v] or "Unknown"
		end
		return r
	end
}

methods["availableTransformations"] = 
{
	"Get a list of operations available",
	"",
	"1 Table of strings: Operation names",
	function(cp)
		local r = {}
		for k,v in pairs(op_names) do
			r[k] = op_names[v]
		end
		return r
	end
}
	




-- inject above into existing metatable for object
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


