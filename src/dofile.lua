local dofile_original = dofile

local internals = {}
local protocol = "maglua://"

function dofile(filename)
	local prefix = string.sub(filename, 1, string.len(protocol))
	local suffix = string.sub(filename, string.len(protocol)+1)
	if prefix == protocol then
		suffix = suffix or ""
		suffix = string.lower(suffix)

                if internals["="..suffix] then
                    return assert(loadstring(internals["="..suffix], "=" .. filename))()
                end

		if internals[suffix] then
			return assert(loadstring(internals[suffix], "=" .. filename))()
		end
		error("Failed to find internal data `" .. (suffix or "") .. "'")
	end
	
	dofile_original(filename)
end

function dofile_add(filename, filecontents)
	internals[string.lower(filename)] = filecontents
end

function dofile_get(filename)
	if filename == nil then
		return internals
	end
	return internals[string.lower(filename)]
end

local io_open = io.open
function io.open(filename, mode)
    local prefix = string.sub(filename, 1, string.len(protocol))
    local suffix = string.sub(filename, string.len(protocol)+1)
    if prefix == protocol then
        suffix = suffix or ""
        suffix = string.lower(suffix)
        if internals[suffix] then
            if mode == "r" or mode == "rb" then
                local f = io.tmpfile()                
                f:write(internals[suffix])
                f:seek("set", 0)
                return f
            end
        end
    end
    
    return io_open(filename, mode)
end

