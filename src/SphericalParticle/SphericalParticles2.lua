-- This script sets up and simulates a spherical particle.
-- See Byron Southern's code and Ken Adebayo's thesis.

--
-- The first part of this script sets up the lattice as a list of
-- locations, types (A1,A1,B1,..) and classifies as core/surf and
-- vacancies.
--
-- The second part translates all site locations so that the smallest
-- coordinate is 1. Neighbours are then defined by offsets.
--
-- With this information of site locations and neighbour lists, the
-- standard maglua initialization starts. An update (step) function is
-- defined and the main loop is entered.
--
-- Note: The exchange strength is set to 1/2 the strength in the MC 
--       code this needs to be examined closely. 

-- parameters from MC code

if arg[1] == nil then
	error("Must supply L as first command line argument")
end

L  = tonumber(arg[1])
RS = 4*L
RC = RS-8

rjaacc=-21*2
rjabcc=-29.1*2
rjbbcc=-8.6*2

rjaacs=-21
rjabcs=-28.1
rjbbcs=-8.6

rjaass=-21/10
rjabss=-28.1/10
rjbbss=-8.6/10

kcore = 0.02
ksurf = 5

nmax, nc, ns, nv = 0, 0, 0, 0

rng = Random.new("Isaac")
sites = {} -- this will hold all site information during initialization

-- given a coordinate, classify site and add to sites table
function checkSite(x, y, z, lbl, occ)
	if x^2 + y^2 + z^2 <= RS^2 then
		local s={
					x=x, y=y, z=z,
					lbl =lbl,
					core=(x^2 + y^2 + z^2) <= RC^2,
					vac =rng:rand() > occ,
					radial = {x,y,z}
				}
				
		table.insert(sites, s)
		
		nmax = nmax + 1
		if s.core then
			nc = nc + 1
		else
			ns = ns + 1
		end
		if s.vac then
			nv = nv + 1
		end
	end
end

for i=-L,L do
	for j=-L,L do
		for k=-L,L do
			-- A1 sites (0,0,0)+fcc
			ax1 = 4*(i+k)
			ay1 = 4*(i+j)
			az1 = 4*(j+k)
			checkSite(ax1, ay1, az1, "A1", 1.0)
			
			-- A1 sites (1/4,1/4,1/4)+fcc
			ax2 = ax1+2
			ay2 = ay1+2
			az2 = az1+2
			checkSite(ax2, ay2, az2, "A2", 1.0)
			
			-- B1 sites (1/8,5/8,1/8)+fcc
			bx1 = ax1 + 1
			by1 = ay1 + 1
			bz1 = az1 - 3
			checkSite(bx1, by1, bz1, "B1", 5/6)
			
			-- B2 sites (3/8,5/8,3/8)+fcc
			bx2 = ax1 + 3
			by2 = ay1 + 1
			bz2 = az1 - 1
			checkSite(bx2, by2, bz2, "B2", 5/6)
			
			-- B3 sites (3/8,7/8,1/8)+fcc
			bx3 = ax1 - 1
			by3 = ay1 + 3
			bz3 = az1 + 1
			checkSite(bx3, by3, bz3, "B3", 5/6)
			
			-- B4 sites (1/8,7/8,3/8)+fcc
			bx4 = ax1 + 1
			by4 = ay1 + 3
			bz4 = az1 - 1
			checkSite(bx4, by4, bz4, "B4", 5/6)
		end
	end
end

print("#sites="..nmax.." #core="..nc.." #surface="..ns.." #vacancies="..nv)

-- These are the offsets to the neighbours for each site type
nn = {}
nn["A1"] = {{ 2, 2, 2}, { 2,-2,-2}, {-2, 2,-2}, {-2,-2, 2}, -- a2
			{ 1, 1,-3}, { 1,-3, 1}, {-3, 1, 1}, -- b1
			{-1, 1, 3}, {-1,-3,-1}, { 3, 1,-1}, -- b2
			{-1,-1,-3}, {-1, 3, 1}, { 3,-1, 1}, -- b3
			{ 1,-1, 3}, { 1, 3,-1}, {-3,-1,-1}} -- b4

nn["A2"] = {{-2,-2,-2}, {-2, 2, 2}, { 2,-2, 2}, { 2, 2,-2}, -- a1
			{-1,-1, 3}, {-1, 3,-1}, { 3,-1,-1}, -- b1
			{ 1,-1,-3}, { 1, 3, 1}, {-3,-1, 1}, -- b2
			{ 1, 1, 3}, { 1,-3,-1}, {-3, 1,-1}, -- b3
			{-1, 1,-3}, {-1,-3, 1}, { 3, 1, 1}} -- b4

nn["B1"] = {{ 2, 1, 2}, {-2, 1,-2}, { 2, 2, 1}, -- b2,b3,b4
			{-2,-2, 1}, { 1, 2, 2}, { 1,-2,-2}, 
			{-1,-1, 3}, {-1, 3,-1}, { 3,-1,-1}, -- a1,a2
			{ 1, 1,-3}, { 1,-3, 1}, {-3, 1, 1}}

nn["B2"] = {{ 2, 1, 2}, {-2, 1,-2}, {-2, 2, 1}, -- b1,b3,b4
			{ 2,-2, 1}, { 1, 2,-2}, { 1,-2, 2},
			{ 1,-1,-3}, { 1, 3, 1}, {-3,-1, 1}, -- a1,a2
			{-1, 1, 3}, {-1,-3,-1}, { 3, 1,-1}}

nn["B3"] = {{ 2, 1,-2}, {-2, 1, 2}, { 2, 2, 1}, -- b1,b2,b4
			{-2,-2, 1}, { 1, 2,-2}, { 1,-2, 2},
			{ 1, 1, 3}, { 1,-3,-1}, {-3, 1,-1}, -- a1,a2
			{-1,-1,-3}, {-1, 3, 1}, { 3,-1, 1}}

nn["B4"] = {{ 2, 1,-2}, {-2, 1, 2}, { 2,-2, 1}, -- b1,b2,b3
			{-2, 2, 1}, { 1, 2, 2}, { 1,-2,-2},
			{-1, 1,-3}, {-1,-3, 1}, { 3, 1, 1}, -- a1,a2
			{ 1,-1, 3}, { 1, 3,-1}, {-3,-1,-1}}

-- shift all sites so they have positive coordinates
minx, miny, minz = 1, 1, 1
maxx, maxy, maxz = 1, 1, 1
for k,v in pairs(sites) do
	minx = math.min(minx, v.x) 
	miny = math.min(miny, v.y) 
	minz = math.min(minz, v.z) 
	
	maxx = math.max(maxx, v.x) 
	maxy = math.max(maxy, v.y) 
	maxz = math.max(maxz, v.z) 
end

for k,v in pairs(sites) do
	sites[k].x = sites[k].x - (minx - 1)
	sites[k].y = sites[k].y - (miny - 1)
	sites[k].z = sites[k].z - (minz - 1)
end

maxx = maxx - minx + 1
maxy = maxy - miny + 1
maxz = maxz - minz + 1



-- calculating neighbour sites for all sites
for i=1,nmax do
	local neighbours = {}
	local s = sites[i]
	local x, y, z = s.x, s.y, s.z
	for k,v in pairs(nn[s.lbl]) do
		table.insert(neighbours, {x+v[1], y+v[2], z+v[3]})
	end
	s.neighbours = neighbours
end


-- mapping sites table to a 3D array (tables of tables of tables)
data = {}
for i=1,maxx do
	data[i] = {}
	for j=1,maxy do
		data[i][j] = {}
	end
end

for i=1,nmax do
	local s = sites[i]
	local x, y, z = s.x, s.y, s.z
	data[x][y][z] = s
end



-- This is where the familiar Maglua script starts:
-- Create data and operator objects
dims = {maxx, maxy, maxz}
ss  = SpinSystem.new(dims)
ex  = Exchange.new(dims)
ani = Anisotropy.new(dims)
th  = Thermal.new(dims)
llg = LLG.new("Quaternion")



-- lookup exchange strengths based on site types and core/surf
function exStr(s1, s2)
	local t1 = string.sub(s1.lbl,1,1) -- first letter of site label
	local t2 = string.sub(s2.lbl,1,1)
	local c1 = s1.core
	local c2 = s2.core
	
	if t1 == "A" and t2 == "A" then
		if c1 ~= c2 then return rjaacs end --c/s
		if c1       then return rjaacc end --c/c
		return rjaass                      --s/s
	end

	if t1 == "B" and t2 == "B" then
		if c1 ~= c2 then return rjbbcs end
		if c1       then return rjbbcc end
		return rjbbss
	end

	if c1 ~= c2 then return rjabcs end 
	if c1       then return rjabcc end
	return rjabss

end

-- does site exist and is not a vacancy?
function real_site(x,y,z)
	if x < 1 or y < 1 or z < 1 then
		return false
	end
	
	if x > maxx or y > maxy or z > maxz then
		return false
	end

	if data[x][y][z] == nil then
		return false
	end
	
	return not (data[x][y][z].vac)
end

-- setup system: initial orientation, exchage & anisotropy
for z=1,maxz do
	for y=1,maxy do
		for x=1,maxx do
			if real_site(x,y,z) then
				local s1 = data[x][y][z]
				local my_type = s1.lbl

				-- initial orientation
				--  site {x,y,z} will point in a random 
				--  direction with unit magnetization
				ss:setSpin({x,y,z}, 
						   {rng:normal(),rng:normal(),rng:normal()}, 1)

				-- setup exchange interaction
				for k,v in pairs(s1.neighbours) do
					local a, b, c = v[1], v[2], v[3]
					if real_site(a,b,c) then
						local s2 = data[a][b][c]
						ex:addPath({x,y,z}, {a,b,c}, exStr(s1, s2))
					end
				end
		
				-- setup anisotropy
				if s1.core then
					ani:setSite({x,y,z}, {0,0,1}, kcore)
				else
					-- radial vector is normalized in the C code
					ani:setSite({x,y,z}, s1.radial, ksurf)
				end
			end
		end
	end
end

-- We will now make explicit lists of sites. These will be used to 
-- collect magnetization stats.

types = {"A1", "A2", "B1", "B2", "B3", "B4"}
regions["core"] = {}
regions["surf"] = {}
for k,v in pairs(types) do
	regions["core_" .. v] = {}
	regions["surf_" .. v] = {}
end
for k,v in pairs(sites) do
	if not v.vac then
		if v.core then
			table.insert(regions["core"], v)
			table.insert(regions["core_" .. v.lbl], v)
		else
			table.insert(regions["surf"], v)
			table.insert(regions["surf_" .. v.lbl], v)
		end
	end
end

mags = {}
for k,v in pairs(regions) do
	mags[k] = {}
end


-- update function
function step()
	ss:zeroFields()
	
	ani:apply(ss)
	 ex:apply(ss)
	 th:apply(rng, ss)
	
	ss:sumFields()
	
	llg:apply(ss)
end

-- We will start at 40, equilibriate for 0.7t, record stats for 0.3t
-- and then reduce temperature by 1T. When temp <6T, the step size will
-- drop to 0.1T
time_per_temp = 2
eq_time       = 0.5
next_step     = 0
temp_step     = -5
last_temp_steps = -1
current_temp  = 100 - temp_step

next_sample   = 0
sample_dt     = 0.01


ss:setTimeStep(0.005)
ss:setAlpha(1.0)
ss:setTime(-2) -- give the system some time to reach equilibrium
-- core/surf net mag and core num samples

f = io.open("mt" .. L .. ".dat", "w")
f:write("# SphericalParticle2.lua script results\n")
f:write("# temp cm sm tm\n")

-- save configuration to file
function report()
-- 	t = string.format("%05.3f", ss:time()) --formatted time
-- 	g = io.open("state-" .. t .. ".dat", "w")
-- 	for z=1,dims[3] do
-- 		for y=1,dims[2] do
-- 			for x=1,dims[1] do
-- 				sx, sy, sz = ss:spin(x,y,z)
-- 				if sx^2 + sy^2 + sz^2 > 1e-4 then
-- 					g:write(table.concat({x, y, z, sx, sy, sz}, "\t") .. "\n")
-- 				end
-- 			end
-- 		end
-- 	end
-- 	g:close()
-- 	print(t)
end

function avrgMag(region)
	local avrg = 0
	local count = 0
	for k,v in pairs(region) do
		avrg = avrg + v
		count = count + 1
	end
	return avrg / count
end


-- Simulation main loop
while current_temp >= 0 do
	local t = ss:time()
	print(t, next_step, current_temp)
	-- check for next temperature step
	if t >= next_step then
		if current_temp < 6 then
			temp_step = last_temp_steps
		end
		
		for k,v in pairs(mags) do
			
		end
		
		if cn > 0 then
			w = {current_temp, cm / cn, sm / sn, (cm+sm) / (cn+sn) }
			line = table.concat(w, "\t")
			f:write(line .. "\n")
			f:flush() -- so we don't have to wait to see the data
			print(line)
			report()
		end

		current_temp = current_temp + temp_step
		th:setTemperature(current_temp)
		-- reset core/surf net mag and core num samples
		cm, cn = 0, 0
		sm, sn = 0, 0
		
		-- let the system adjust to the new temp
		next_sample= t + eq_time
		next_step  = t + time_per_temp
	end

	step() -- do 1 LLG step
	
	-- check for sampling step
	if t >= next_sample then
		for k1,region in pairs(regions) do
			for k2,site in pairs(region) do
				local sx, sy, sz = ss:spin(site.x, site.y, site.z)
				table.insert(mags[k1], (sx^2 + sy^2 + sz^2)^(1/2))
			end
		end
		next_sample = ss:time() + sample_dt
	end
end
f:close()