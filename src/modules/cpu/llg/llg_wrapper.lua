-- This is a wrapper to allow the new LLG.Type.new() style
-- to be called as LLG.new("Type")

if LLG.new == nil then
	LLG.new = function(t, ...)
		if LLG[t] == nil then
		    error("Failed to find LLG implementation of type `" .. tostring(t) .. "'")
		end
		
		return LLG[t].new(...)
	end
end
