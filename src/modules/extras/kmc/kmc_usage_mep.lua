-- This is a template showing how to implement an MEP core for the KMC framework

ss = SpinSystem.new(8,8,2)

ex = Exchange.new(ss)
ani = Anisotropy.new(ss)
thermal = Thermal.new(ss, Random.Isaac.new())
zee = AppliedField.new(ss)

zee:set({0,0,-1})

mep = MEP.new()

ex:setPeriodicXYZ(true, true, false)
for pos,mag in ss:eachSite() do
	local i,j,k = pos[1], pos[2], pos[3]
	
	ss:setSpin(pos, {0,0,1})

	ex:add({i+1,j,k}, pos, 0.1)
	ex:add({i-1,j,k}, pos, 0.1)
	ex:add({i,j+1,k}, pos, 0.1)
	ex:add({i,j-1,k}, pos, 0.1)
	ex:add({i,j,k+1}, pos, 2)
	ex:add({i,j,k-1}, pos, 2)

	ani:add(pos, {0,0,1}, 4)
end

-- function that calculates the energy of a SpinSystem
function mep_energy(ss)
	ss:resetFields()
	ex:apply(ss)
	ani:apply(ss)
	zee:apply(ss)
	ss:sumFields()

        return -1 * (
			ss:fieldArrayX("Total"):dot(ss:spinArrayX()) +
                ss:fieldArrayY("Total"):dot(ss:spinArrayY()) +
                ss:fieldArrayZ("Total"):dot(ss:spinArrayZ()) )
end

mep:setEnergyFunction(mep_energy)
mep:setSpinSystem(ss)
mep:setTolerance(1e-1)
mep:setCoordinateSystem("Spherical")

kmc = KMC.new() -- empty framework, needs some functions and info
kmc:setFields({ex,ani,thermal,zee})

-- We will be considering complete columns in the KMC
-- so we need to define a custom position that doesn't
-- consider the layers separately. Also we need to think
-- about the different switching mechanisms. This means our 
-- position will be a 3 component value: {x, y, "switch type"}
-- where "switch type" is defines as:
--  1: coherent rotation
--  2: top switches only
--  3: bottom switches only
-- a coherent rotation will not always be possible as our starting
-- state may not be a coherent configuration. 
-- 
-- Let's build the 1D list of custom positions:
local positions = {}
for x=1,ss:nx() do
	for y=1,ss:ny() do
		for t=1,3 do
			table.insert(positions, {x,y,t})
		end
	end
end
-- and tell the KMC object to use this list
kmc:setPositions(positions) -- no longer iterating over SpinSystem


-- now, for each position we need to compute the energy barrier, and
-- attempt frequency for a given action
function ebaf(kmc, ss, pos)
	-- here are the sites that will be involved:
	local sites = {{pos[1], pos[2], 1}, {pos[1], pos[2], 2}}
	local num_path_points = 12 -- we will define our line with this number of points

	-- the start point for a path is the current configuration of the sites
	local mx1, my1, mz1 = ss:spin(sites[1])
	local mx2, my2, mz2 = ss:spin(sites[2])
	local path = {}
	path[1] = {{mx1,my2,mz2}, {mx2,my2,mz2}}

	-- the next point in the path depends on pos[3]
	if pos[3] == 1 then -- coherent rotation
		-- Before we do anything in this case we need to 
		-- make sure our starting condition is valid:
		-- Both layers are pointing in the same direction
		if (mx1*mx2 + my1*my2 + mz1*mz2) < 0 then --dot product is negative
			return -- returning nothing (ie nil), this means we wont consider this case
		end
		-- tests passed, we will now define the final orientations we are interested in
		path[2] = {{mx1,my2, -mz2}, {mx2,my2, -mz2}}
	end

	-- these cases have no start requirements
	if pos[3] == 2 then -- top switches only
		path[2] = {{mx1,my2,-mz2}, {mx2,my2, mz2}}
	end
	if pos[3] == 3 then -- bottom switches only
		path[2] = {{mx1,my2, mz2}, {mx2,my2,-mz2}}		
	end

	mep:setSites(sites)
	mep:setInitialPath(path)
	mep:setNumberOfPathPoints(2)
	-- The final site may not be at the bottom of the well so
	-- we will relax it. Relaxing means a drop in energy which 
	-- takes no time so this is valid to do in here
	mep:relaxSinglePoint(2, 10) -- relaxing point 2 for 10 adaptive iterations 

	-- now we have a better final point, time to interpolate entire path
	mep:setNumberOfPathPoints(num_path_points)

	mep:compute(20) -- run 20 adaptive steps, should get something close

	-- now that we have something close we will relax the maximal points to get 
	-- a more accurate energy barrier
	local mins, maxs, all = mep:maximalPoints()
	local tol = mep:tolerance()
	mep:setTolerance(tol/10)
	for k,pidx in pairs(mins) do
        mep:relaxSinglePoint(pidx, 10)
	end
	for k,pidx in pairs(maxs) do
        mep:relaxSaddlePoint(pidx, 10)
	end
	mep:setTolerance(tol)

	local eb = mep:energyBarrier()


	-- time to calculate the attempt frequency
	local f0 = 1 + 0 * (math.exp(math.pi) + 42)
	f0 = f0 * 1e50

	-- and create a function that describes the motion if this were to happen
	local sx1, sy1, sz1, sm1 = mep:spin(num_path_points, 1)
	local sx2, sy2, sz2, sm2  = mep:spin(num_path_points, 2)
	local function event(ss)
		ss:setSpin(sites[1], {sx1,sy1,sz1}, sm1)
		ss:setSpin(sites[2], {sx2,sy2,sz2}, sm2)
	end


	local kT = kmc:temperature()
	local rng = kmc:rng()
	local decay = f0 * math.exp(-eb/kT)
	local r = rng:rand()

	local time = -math.log(r) / decay


	return time, event
end

kmc:setEventFunction(ebaf)

thermal:setTemperature(0.2)
-- We could have used kmc:setTemperature(0.2)

-- main loop:
while ss:time() < 10 do
	kmc:apply(ss) -- this call is internally parallelized via MPI
end

