local dofile_original = dofile

local internals = {}
local protocol = "maglua://"
function dofile(filename)
	
	local prefix = string.sub(filename, 1, string.len(protocol))
	local suffix = string.sub(filename, string.len(protocol)+1)
	if prefix == protocol then
		if internals[suffix] then
			return assert(loadstring(internals[suffix]))()
		end
		error("Failed to find internal data `" .. suffix or "" .. "'")
	end
	
	dofile_original(filename)
end

function dofile_add(filename, filecontents)
	internals[filename] = filecontents
end

function dofile_get(filename)
	if filename == nil then
		return internals
	end
	return internals[filename]
end

