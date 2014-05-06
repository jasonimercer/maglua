-- CUrl

local orig_dofile = dofile

function dofile(x)
	local a, b = curl(x)
	
	if a == false then -- try the default?
		return orig_dofile(x)
	end

	return assert(loadstring(a, "=" .. x))()
end
