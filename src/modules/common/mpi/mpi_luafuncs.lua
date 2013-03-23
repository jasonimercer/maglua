-- MPI support functions

local MODNAME = "mpi"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

	

mpi.comm_world = mpi.get_comm_world()
mpi.get_comm_world = nil
	
-- make an iterator that iterates over different balanced 
-- chunks of [a:step:b] for each MPI process
t._make_range_iterator = function(comm, a, b, step)
	local size = comm:get_size()
	local rank = comm:get_rank()
	
	step = step or 1
	local elements = 0
	local t = {}
	for i=a,b,step do
		table.insert(t, i)
		elements = elements + 1
	end
	
	if elements == 0 then
		return function() return nil end
	end
	
	local l = elements % size
	local r = math.floor(elements / size)
	local v1 = 1
	local v2
	
	local mya, myb
	
	for i=1,size do
		v2 = v1 + r
		
		if i > l then
			v2=v2-1
		end

		if i == rank then
			mya, myb = v1, v2
		end
		v1 = v2+1;
	end
	
	if mya == nil then
		return function() return nil end
	end
	
	if t[mya] and t[myb] == nil then --single value
		local function f()
			local value = t[mya]
			return function()
				local s = value
				value = nil
				return s
			end
		end
		return f()
	end

	local function f()
		local value = t[mya]
		local endval = t[myb]
		local stepsize = step
		
		return function()
			if value == nil then
				return value
			end
			local return_value = value
			if math.abs(value - endval) < step*0.1 then
				value = nil
			else
				value = value + stepsize
			end
			return return_value
		end
	end
	return f()
end
	

mpi._make_range_iterator = t._make_range_iterator
--[[
local help = MODTAB.help

MODTAB.help = function(x)
-- 	if x == t._make_range_iterator then
-- 		return "","",""
-- 	end

-- 	if x == saveTensors then
-- 		return
-- 		"Convenience method to save the tensors in a lua readable file",
-- 		"2 Strings: The filename used for writing the tensor file, traditionally ending with a .lua extension. Optional note to put in the file.",
-- 		""
-- 	end
-- 	
-- 	if x == loadTensors then
-- 		return
-- 		"Convenience method to load interaction tensors from a file",
-- 		"1 String: The filename of the tensor file.",
-- 		""
-- 	end
	
	if x == nil then
		return help()
	end
	return help(x)
end]]

