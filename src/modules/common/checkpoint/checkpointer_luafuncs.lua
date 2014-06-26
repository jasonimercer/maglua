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

methods["addTable"] =
{
	"Add each element in a table to the Checkpointer",
	"1 Table: Each value will be added",
	"",
	function(cp, t)
		for i=1,table.maxn(t) do
			cp:add(t[i])
		end
	end
}

methods["getTable"] =
{
	"Get all values in Checkpointer, put them in a table and return them.",
	"",
	"1 Table: All values in Checkpointer",
	function(cp)
		return cp:_getTable()
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
	"0 or more Strings. Apply the given operation names, case does not matter.",
	"",
	function(cp, ...)
		for j=1,table.maxn(arg) do
			local op_name = arg[j] or ""
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
local mod_help = MODTAB.help

-- create new help function for methods above
MODTAB.help = function(x)
    for k,v in pairs(methods) do
        if x == v[4] then
            return v[1], v[2], v[3]
        end
    end

    -- fallback to old help if a case is not handled
    if x == nil then
        return mod_help()
    end
    return mod_help(x)
end



dofile("maglua://Help.lua") --get default global scope help function


local global_help = help


function checkpointSave(fn, ...)
	local cp = Checkpointer.new()
	cp:addTable(arg)
	cp:transform("Pack") --, "Base64")
	cp:saveToFile(fn)
end

function checkpointLoad(fn)
	local cp = Checkpointer.new()
	cp:loadFromFile(fn)

	while cp:_transformation() > 0 do
		cp:untransform()
	end

	return cp:get()
end

function checkpointToString(...)
	local cp = Checkpointer.new()
	cp:_setDebugFile("cp2s.txt")
	cp:addTable(arg)
	cp:transform("Pack", "Base64")
	return cp:toString()
end

function checkpointFromString(txt)
	local cp = Checkpointer.new()
	cp:fromString(txt)

	while cp:_transformation() > 0 do
		cp:untransform()
	end

	return cp:get()
end



local function chop_string(x, n, t)
	t = t or {}
	n = n or 80
	if string.len(x) <= n then
		table.insert(t, x)
		return t
	end
	local a, b = string.sub(x, 1, n), string.sub(x, n+1)
	table.insert(t, a)
	return chop_string(b, n, t)
end



local _f = assert(loadstring("return function(x) return x*x end"))()
--[[
function f(x)
	return x*x
end
--]]
function help(x)
	local f = _f
	if x == nil then
		return global_help()
	end

	
	if x == checkpointSave then
		return 
[[Function to save zero or more values to a checkpoint file. Example:
<pre>
function f(x)
  return x*x
end

checkpointSave("checkpoint_help_example.dat", f, 5, {"a", "b"})</pre>
]],
    "1 String, 0 or more Values: The String is the file name to be used, values will be encoded and saved to file",
	""
	end

	if x == checkpointLoad then
		return 
			[[Function to load zero or more values from a checkpoint file. Example: <pre>
g, value, t = checkpointLoad("checkpoint_help_example.dat")
print(g(value), t[1], t2])
</pre>]],
"1 String: The String is the file to be read",
"0 or more Values: The values encoded in the file are returned from the function."
	end


	local ss = checkpointToString(f, 5, {"a", "b"})
	
	if x == checkpointToString then

		return 
[[Function to save zero or more values to a checkpoint string encoded using the uuencode algorithm. Encoded data use only printable characters and so can be easily emailed, copied to a clipboard, stored in a text file or stored in a database.
Example:<pre>
function f(x)
  return x*x
end

s = checkpointToString(f, 5, {"a", "b"})
print(s)</pre>Expected Output (chopped into lines of 60 characters for readability):<pre>]] .. table.concat(chop_string(ss, 60), "\n") .. [[</pre>]],
"0 or more Values: Values will be encoded and returned as a string",
"1 String: The string representing the encoded values as a printable string"
	end

	if x == checkpointFromString then
		local r1,r2,r3 = checkpointFromString(ss)

		local statement = "s = \"" .. table.concat(chop_string(ss, 60), "\" ..\n    \"") .. "\"\n"
		--[[
		local res = r1(r2)

		local r = debug.getinfo( r1 )
		res = res .. "\n"
		for k,v in pairs(r) do
			res = res .. "\n" .. k .. ":" .. tostring(v)
		end

		r1 = tostring(r1)
		r2 = tostring(r2)
		r3 = tostring(r3)

		local r = r1 .. " " .. r2 .. " " .. r3 .. ": " .. res
			--]]

		return "Function to load zero or more values from a checkpoint string. Encoded data use only printable characters and so can be easily emailed, copied to a clipboard, stored in a text file or stored in a database. Example:<pre>" .. statement .. "\nf, v, t = checkpointFromString(s)\n</pre>",
        "1 String: The string representing the values as a printable string.",
		"0 or more Values: Values will be decoded and returned." 
	end


	return global_help(x)
end


_f = nil
