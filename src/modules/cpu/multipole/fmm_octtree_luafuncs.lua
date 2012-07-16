-- fmm_octtree_luafuncs.lua
-- 
-- these are additional methods implemented in lua for the FMMOctTree
-- 
-- 
local MODNAME = "FMMOctTree"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local function no_checkRunAtLeaf(t, f, ...)
	if t:child(1) then
		for i=1,8 do
			no_checkRunAtLeaf(t:child(i), f, ...)
		end
	else
		f(t, ...)
	end
end

local function runAtLeaf(t, f, ...)
	if type(f) ~= "function" then
		error("First argument of `runAtLeaf' must be a function")
	end
	
	no_checkRunAtLeaf(t, f, ...)
end



local function no_checkRunAtPopulatedLeaf(t, f, ...)
	if t:count() > 0 then
		if t:child(1) then
			for i=1,8 do
				no_checkRunAtPopulatedLeaf(t:child(i), f, ...)
			end
		else
			f(t, ...)
		end
	end
end
local function runAtPopulatedLeaf(t, f, ...)
	if type(f) ~= "function" then
		error("First argument of runAtPopulatedLeaf' must be a function")
	end
	
	no_checkRunAtPopulatedLeaf(t, f, ...)
end

local function findNodeAtPosition(t, pos)
	if not t:contains(pos) then
		return nil
	else
		if t:child(1) then
			for i=1,8 do
				if t:child(i):contains(pos) then
					return findNodeAtPosition(t:child(i), pos)
				end
			end
		else
			return t
		end
	end
end

local function dimensionsOfNChild(t, n)
	local xyz = t:dimensions()
	
	local x,y,z = xyz[1], xyz[2], xyz[3]
	
	while n > 0 do
		x = x * 0.5
		y = y * 0.5
		z = z * 0.5
		n = n - 1
	end
	
	while n < 0 do
		x = x * 2
		y = y * 2
		z = z * 2
		n = n + 1
	end

	return {x,y,z}
end


local function children(t)
	local cc = {}
	for i=1,8 do
		cc[i] = t:child(i)
	end
	return cc
end

local function toGlobalCoordinates(octtree, coord)
	local o = octtree:localOrigin()
	if type(coord) == "table" then
		return {coord[1] + o[1], coord[2] + o[2], coord[3] + o[3]}
	end
	return nil
end

local function toLocalCoordinates(octtree, coord)
	local o = octtree:localOrigin()
	if type(coord) == "table" then
		return {coord[1] - o[1], coord[2] - o[2], coord[3] - o[3]}
	end
	return nil
end

local function buildRelationships(octtree)
	local t = octtree
	while t:child(1) do
		t = t:child(1)
	end
	leaf_dims = t:dimensions()

	local function addNearAtDelta(t, d)
		-- find node at given delta
		local o = t:localOrigin()
		local n = octtree:findNodeAtPosition( t:toGlobalCoordinates(d) ) --might not exist. That's OK. 
		
		local ed = t:extraData() or {}
		ed.near = ed.near or {}
		
		table.insert(ed.near, n) --might add nil, no prob.
		
		t:setExtraData(ed)
	end

	-- now we will decend through the entire tree and for each leaf we'll find the near
	-- neighbours. 
	octtree:runAtPopulatedLeaf(addNearAtDelta, { 0,0,0})
	octtree:runAtPopulatedLeaf(addNearAtDelta, { leaf_dims[1],0,0})
	octtree:runAtPopulatedLeaf(addNearAtDelta, {-leaf_dims[1],0,0})
	octtree:runAtPopulatedLeaf(addNearAtDelta, {0, leaf_dims[2],0})
	octtree:runAtPopulatedLeaf(addNearAtDelta, {0,-leaf_dims[2],0})
	octtree:runAtPopulatedLeaf(addNearAtDelta, {0,0, leaf_dims[3]})
	octtree:runAtPopulatedLeaf(addNearAtDelta, {0,0,-leaf_dims[3]})


	-- find nodes closest (generation) to the root that don't contain the near nodes
	local function addFarNotNear(t)
		local function findFar(t, near, far)
			local OK = true
			for k,v in pairs(near) do
				if t:contains(v) then
					OK = false
				end
			end
			
			if OK then
				table.insert(far, t)
			else
				if t:child(1) then
					for i=1,8 do
						findFar(t:child(i), near, far)
					end
				end
			end
		end
		
		local ed = t:extraData()
		local near = ed.near
		local far = {}
		
		findFar(octtree, near, far) -- octtree is the root
		
		ed.far = far
		t:setExtraData(ed)
	end

	-- and then using the near nodes, find the far ones (which exclude near)
	octtree:runAtPopulatedLeaf(addFarNotNear)

	-- take the lua extra data and push it into the C++ world
	function inject_nearfar(node)
		local t = node:extraData() or {}

		for k,v in pairs(t.near or {}) do
			node:addNear(v)
		end
		for k,v in pairs(t.far or {}) do
			node:addFar(v)
		end
	end
	octtree:runAtLeaf(inject_nearfar)
end




t.runAtLeaf = runAtLeaf
t.runAtPopulatedLeaf = runAtPopulatedLeaf
t.findNodeAtPosition = findNodeAtPosition
t.dimensionsOfNChild = dimensionsOfNChild
t.children = children
t.buildRelationships = buildRelationships
t.toGlobalCoordinates = toGlobalCoordinates
t.toLocalCoordinates = toLocalCoordinates

local help = MODTAB.help

MODTAB.help = function(x)

	if x == toGlobalCoordinates then
		return
		"Convenience method to convert a position {x,y,z} from local coordinates to global coordinates",
		"1 Table: {x,y,z} of point to convert",
		"1 Table: {x,y,z} of point in global coordinates"
	end
	
	if x == toLocalCoordinates then
		return
		"Convenience method to convert a position {x,y,z} from global coordinates to local coordinates",
		"1 Table: {x,y,z} of point to convert",
		"1 Table: {x,y,z} of point in local coordinates"
	end
	
	if x == runAtLeaf then
		return
		"Convenience method to evaluate a function at all leaf nodes",
		"1 Function, 0 or more arguments: The function to run and any required arguments. The leaf will be given as the 1st argument when the function is called.",
		""
	end
	
	if x == buildRelationships then
		return
		"Build near and far node relationships. Required before any field calculation calls.",
		"",
		""
	end
	
	if x == runAtPopulatedLeaf then
		return
		"Convenience method to evaluate a function at all populated leaf nodes",
		"1 Function, 0 or more arguments: The function to run and any required arguments. The leaf will be given as the 1st argument when the function is called.",
		""
	end
	
	if x == findNodeAtPosition then
		return
		"Find the deepest FMMOctTree node that contains the gven position",
		"1 position: The position given must be compatible with the contains function.",
		"0 or 1 FMMOctTree: the node that contains the point."
	end

	if x == dimensionsOfNChild then
		return 
		"Calculate the dimensions of child that would be n generations below the given node.",
		"1 integer: The number of generations below the given node (can be negative for parents)",
		"1 Table of 3 numbers: The x, y and z dimensions of the node at the generation offset."
	end
	
	
	if x == children then
		return 
		"Get all children of the node in a table with keys 1 through 8.",
		"",
		"1 Table: Children of given node, children may be nil for leav node."
	end
	
	if x == nil then
		return help()
	end
	return help(x)
end


