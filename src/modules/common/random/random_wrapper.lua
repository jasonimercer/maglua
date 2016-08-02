-- This is a wrapper to allow the new Random.Type.new() style
-- to be called as Random.new("Type")

if Random.new == nil then
	Random.new = function(t, ...)
		if Random[t] == nil then
			error("Failed to find random number generator of type `" .. t .. "'")
		end
		
		return Random[t].new(...)
	end
end
