ss = SpinSystem.new(2,1)

dd = DisorderedDipole.new(ss)

max_degree = 5
epsilon = 1e-5
local nsplit = 2


f = "Exchange" -- fmm field in Exchange for testing

c = 1
for j=1,ss:ny() do
	for i=1,ss:nx() do
		ss:setSpin({c}, {1,0,0}, 1)
		dd:setSitePosition({c}, {i,j,0}) --this also sets oct tree sites
		c = c + 1
	end
end
-- ss:setSpin({1}, {0,1,0}, 1)

-- linking fmm positions to dd positions
ot = FMMOctTree.new(max_degree,  dd:arrayX(),       dd:arrayY(),       dd:arrayZ(),
								 ss:spinArrayX(),   ss:spinArrayY(),   ss:spinArrayZ(), 
								 ss:fieldArrayX(f), ss:fieldArrayX(f), ss:fieldArrayX(f) ) 



ot:split(nsplit) --split volume N times
print("ABC")
ot:buildRelationships() -- req'd before field calcs
print("DEF")

local hx = ss:fieldArrayX(f)
local hy = ss:fieldArrayY(f)
local hz = ss:fieldArrayZ(f)

local sx = ss:spinArrayX()
local sy = ss:spinArrayY()
local sz = ss:spinArrayZ()

local px = dd:arrayX()
local py = dd:arrayY()
local pz = dd:arrayZ()

function calculateFields(node)
	local near = node:extraData()["near"]
	local far  = node:extraData()["far"]

	local function dipField(mx, my, mz, rx, ry, rz)
		local r = (rx^2 + ry^2 + rz^2)^(1/2)
		print("dip field")
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
		for k,f in pairs(far) do
			print("far",k)
			local o = f:localOrigin()
			local x = px:get(m) - o[1]
			local y = py:get(m) - o[2]
			local z = pz:get(m) - o[3]
			
			local hhx, hhy, hhz = f:fieldAt(x,y,z)
			local hhx, hhy, hhz = f:fieldAt( f:toLocalCoordinates( {pxm, pym, pzm} ) )
			print(hhx, hhy, hhz)
			
			hx:addAt(m, hhx)
			hy:addAt(m, hhy)
			hz:addAt(m, hhz)
		end

		
		-- interact with all near nodes (which includes self)
		for k,n in pairs(near) do
			for i=1,n:count() do
				local q = n:member(i)
				if q ~= m then
					local pxq = px:get(q)
					local pyq = py:get(q)
					local pzq = pz:get(q)
					
					local dhx, dhy, dhz = dipField(sx:get(q), sy:get(q), sz:get(q), pxm-pxq, pym-pyq, pzm-pzq)
					print("AAA", dhx, dhy, dhz)
					hx:addAt(m, dhx)
					hy:addAt(m, dhy)
					hz:addAt(m, dhz)
				end
			end
		end
	end
end

function fields()
	ss:resetFields()
	
	ot:calcInnerTensor(epsilon)
	--ot:calculateDipoleFields()
	ot:runAtPopulatedLeaf(calculateFields)

	ss:setSlotUsed(f, true)

	dd:apply(ss)
	
	
	ss:sumFields()
	
	--llg:apply(ss)
end


fields()

pos = {1,1}
print("Brute Force:")
print(ss:field("Dipole", pos))
print("FMM:")
print(ss:field("Exchange", pos))
