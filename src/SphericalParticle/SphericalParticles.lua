-- This script sets up and simulates a simplified spherical particle.
-- Surface anisotropy points radially away from center. Inner anisotropy
-- points along {0,0,1}.

radius = 8
n = radius * 2 + 2
cx, cy, cz = radius, radius, radius


J = 1 --exchange strength
K_core = 1 -- core anisotropy
K_surf = 2 -- surface anisotropy

nxyz = {n,n,n}

-- setup data and operators
ss  = SpinSystem.new(nxyz)
ex  = Exchange.new(nxyz)
ani = Anisotropy.new(nxyz)
th  = Thermal.new(nxyz)
rng = Random.new("Isaac")
llg = LLG.new("Quaternion")

ss:setTimeStep(0.01)
ss:setAlpha(0.1)


-- deterermine if a site is part of the sphere and
-- if it is on the surface, it is on the surace if
-- it is part of the sphere and a neighbour is not
function site_type(x,y,z)
	local function part(x,y,z)
		local r = (x^2 + y^2 + z^2)^(1/2)
		return r < radius + 0.1
	end

	if part(x,y,z) then
		local on_surface = false --assume core
		for dx=-1,1 do
			for dy=-1,1 do
				for dz=-1,1 do
					if not part(x+dx, y+dy, z+dz) then
						on_surface = true
					end
				end
			end
		end
		return true, on_surface
	else
		return false, false
	end
end

-- setup system: initial orientation, exchage & anisotropy
nn_dir = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}}
for z=1,n do
	for y=1,n do
		for x=1,n do
			local part, surf = site_type(x-cx, y-cy, z-cz)
			
			if part then
				ss:setSpin({x,y,z}, {0,0,1})
				-- exchange interaction between sites
				for k,v in pairs(nn_dir) do
					ex:addPath({x,y,z}, {x+v[1],y+v[2],z+v[3]}, J)
				end
				
				if surf then
					ani:setSite({x,y,z}, {x-cx, y-cy, z-cz}, K_surf)
				else
					ani:setSite({x,y,z}, {0,0,1}, K_core)
				end
			end
		end
	end
end


function step()
	ss:zeroFields()
	
	ani:apply(ss)
	 ex:apply(ss)
	 th:apply(rng, ss)
	
	ss:sumFields()
	
	llg:apply(ss)
end

function report()
	t = string.format("%05.3f", ss:time()) --formatted time
	f = io.open("time-" .. t .. ".dat", "w")
	for z=1,nxyz[3] do
		for y=1,nxyz[2] do
			for x=1,nxyz[1] do
				if x < radius or y < radius or z < radius then
					sx, sy, sz = ss:spin(x,y,z)
					f:write(table.concat({x, y, z, sx, sy, sz}, "\t") .. "\n")
				end
			end
		end
	end
	f:close()
	print(t)
end

max_time = 10
while ss:time() < max_time do
	
	local t = ss:time()

	if t > 8 then
		th:setTemperature(0)
	else
		th:setTemperature((max_time - t)/10)
	end

	
	step()
end

report()
