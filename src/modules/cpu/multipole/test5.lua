-- n = arg[1] or 100
-- nx = 10
-- ny = 10
-- n = nx*ny
n = 64
ss = SpinSystem.new(n)
llg = LLG.Cartesian.new()

ss:setTimeStep(5e-3)

max_degree = 5
epsilon = 1e-5

-- these are the "position" arrays. They exist outside of the spinsystem
px = Array.Double.new(n)
py = Array.Double.new(n)
pz = Array.Double.new(n)

hx = ss:fieldArrayX("Dipole")
hy = ss:fieldArrayY("Dipole")
hz = ss:fieldArrayZ("Dipole")

sx = ss:spinArrayX()
sy = ss:spinArrayY()
sz = ss:spinArrayZ()

function r()
	return math.random()*2-1
end
function rrr()
	return {r(), r(), 5+r()}
end

px:set(1,0)
py:set(1,0)
pz:set(1,0)

ss:setSpin({1}, {0,-1,0}, 5)
for i=2,n do
	ss:setSpin({i}, rrr(), 1/2)
	
	local t = math.pi*2.0*(i-2)/(n-1)
	px:set(i, 3.5*math.cos(t))
	py:set(i, 3.5*math.sin(t))
	pz:set(i, math.mod(i,2))
end

-- c = 1
-- for i=2,n do
-- for i=1,ny do
-- 	for j=1,nx do
-- 		px:set(c, i+r()*0.25)
-- 		py:set(c, j+r()*0.25)
-- 		pz:set(c, 0+r()*0.25)
-- -- 		pz:set(c, 0)
-- 		c = c + 1
-- 	end
-- end

function maxGeneration(node)
	if node:child(1) then
		local max = node:child(1):generation()
		for i=2,8 do
			local m = maxGeneration(node:child(i))
			if m > max then
				max = m
			end
		end
		return max
	end
	return node:generation()
end





octtree = FMMOctTree.new(max_degree,  px,py,pz, sx,sy,sz,  hx,hy,hz)
-- octtree:setBounds({-5,-5,-5}, {5,5,5}) -- this call is not needed but useful for testing/experiment
local nsplit = 0
octtree:split(nsplit) --split volume N times


leaf_dims = octtree:dimensionsOfNChild(nsplit)

-- for k,v in pairs(getmetatable(octtree)) do
-- 	print(k,v)
-- end

-- error("AAA")


-- print("Leaf dims:", table.concat(leaf_dims, ", "))


function addNearAtDelta(t, d)
	-- find node at given delta
	local o = t:localOrigin()
	local n = octtree:findNodeAtPosition({o[1] + d[1], o[2] + d[2], o[3] + d[3]}) --might not exist. That's OK. 
	
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


function addFarNotNear(t)
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
	
	findFar(octtree, near, far)
	
	ed.far = far
	t:setExtraData(ed)
end

-- and then using the near nodes, find the far ones (which exclude near)
octtree:runAtPopulatedLeaf(addFarNotNear)


if false then
	leaves = {}
	function findLeafNodes(t)
		table.insert(leaves, t)
	end
	octtree:runAtPopulatedLeaf(findLeafNodes)

	print("num pop'd leaves: ", table.maxn(leaves))



	leaves = {}
	function findLeafNodes(t)
		table.insert(leaves, t)
	end
	octtree:runAtLeaf(findLeafNodes)

	print("num total leaves: ", table.maxn(leaves))
end


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


function calculateFields(node)
	local near = node:extraData()["near"]
	local far  = node:extraData()["far"]

	local function dipField(mx, my, mz, rx, ry, rz)
		local r = (rx^2 + ry^2 + rz^2)^(1/2)
		if r == 0 then
			error("R = 0 in dipField")
		end
		local mdotr = mx*rx + my*ry + mz*rz
		
		return  -mx/(r^3) + 3*rx*mdotr/(r^5),
				-my/(r^3) + 3*ry*mdotr/(r^5),
				-mz/(r^3) + 3*rz*mdotr/(r^5)
	end
	
	for j=1,node:count() do
		local m = node:member(j)
		hx:set(m, 0)
		hy:set(m, 0)
		hz:set(m, 0)
		
		local pxm = px:get(m)
		local pym = py:get(m)
		local pzm = pz:get(m)
		
		-- iterate over far nodes, cast local position in far's coordinates
		-- can compute field
		timer_far:start()
		for k,f in pairs(far) do
			local o = f:localOrigin()
			local x = px:get(m) - o[1]
			local y = py:get(m) - o[2]
			local z = pz:get(m) - o[3]
			
			local hhx, hhy, hhz = f:fieldAt(x,y,z)
			
			hx:addAt(m, hhx)
			hy:addAt(m, hhy)
			hz:addAt(m, hhz)
		end
		timer_far:stop()
		
		timer_near:start()
		-- interact with all near nodes (which includes self)
		for k,n in pairs(near) do
			for i=1,n:count() do
				local q = n:member(i)
				if q ~= m then
					local pxq = px:get(q)
					local pyq = py:get(q)
					local pzq = pz:get(q)
					
					local dhx, dhy, dhz = dipField(sx:get(q), sy:get(q), sz:get(q), pxm-pxq, pym-pyq, pzm-pzq)
					
					hx:addAt(m, dhx)
					hy:addAt(m, dhy)
					hz:addAt(m, dhz)
				end
			end
		end
		
		timer_near:stop()
	end
end

timer_far = Timer.new()
timer_near = Timer.new()



function step()
-- 	print("Step")
	ss:resetFields()
	
	octtree:calcInnerTensor(epsilon)
-- 	octtree:runAtPopulatedLeaf(calculateFields)
	octtree:calculateDipoleFields()

	ss:setSlotUsed("Dipole", true)
	ss:sumFields()
	
	llg:apply(ss)
end



timer = Timer.new()
timer:start()
for i=1,5000 do
	print(i)
	step()
end

print(ss:field("Total", {5}))
timer:stop()


print(n, timer:elapsed())--, timer_near:elapsed(), timer_far:elapsed())






checkpointSave("final.dat", ss, px, py, pz)






