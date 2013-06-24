-- MEP

local MODNAME = "MEP"
local MODTAB = _G[MODNAME]
local mt = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local help = MODTAB.help -- setting up fallback to C++ help defs

local function get_mep_data(mep)
	if mep == nil then
		return {}
	end
	if mep:getInternalData() == nil then
		mep:setInternalData({})
	end
	return mep:getInternalData()
end

local function getStepMod(tol, err, maxMotion)
	if maxMotion then
		if maxMotion > 0.1 then
			return 0.5, false
		end
	end
	if err == 0 then
		return 2, true
	end

	if err<tol then
		local x = 0.9*(tol / err)^(0.5)
-- 		print(x)
		return x, err<tol
	end
	
	return 0.95 * (tol / err)^(0.9), err<tol
end

local function getSpinSystem(mep)
	local d = get_mep_data(mep)
	return d.ss
end

local function getTolerance(mep)
	local d = get_mep_data(mep)
	return d.tol or 1e-4
end
local function setTolerance(mep, tol)
	local d = get_mep_data(mep)
	d.tol = tol
end

local function getEnergyFunction(mep)
	local d = get_mep_data(mep)
	return (d.energy_function or function() return 0 end)
end

-- write a state to a spinsystem
local function writePathPointTo(mep, path_point, ssNew)
	local d = get_mep_data(mep)
	local ss = d.ss

	ss:copySpinsTo(ssNew)
-- 	print("pp", path_point)
	local sites = mep:sites()
	for i=1,table.maxn(sites) do
		local x,y,z = mep:spin(path_point, i)
		print(table.concat(sites[i], ","))
		ssNew:setSpin(sites[i], {x,y,z})
	end
end

local function pathEnergy(mep)
	local d = get_mep_data(mep)
	local ss = d.ss
	local energy_function = d.energy_function

	if ss == nil then
		error("Initial State required for pathEnergies")
	end
	if energy_function == nil then
		error("Energy Function required for pathEnergies")
	end

	local function get_site_ss(x,y,z)
		local sx,sy,sz = ss:spin(x,y,z)
		return sx,sy,sz
	end
	local function set_site_ss(x,y,z,sx,sy,sz)
		local _,_,_,m = ss:spin({x,y,z})
		ss:setSpin({x,y,z}, {sx,sy,sz},m)
	end
	local function get_energy_ss()
		return energy_function(ss)
	end

	mep:calculateEnergies(get_site_ss, set_site_ss, get_energy_ss);
	return mep:getPathEnergy()
end

local function randomize(mep, magnitude)
	mep:_randomize(magnitude)
end

local function copy_to_children(mep)
	local d = get_mep_data(mep)
	if not mep:isChild() then
		mep:internalCopyTo(d.big_step)
		mep:internalCopyTo(d.small_step)

		for _,c in pairs({d.big_step, d.small_step}) do
			local id ={}
			for k,v in pairs(d) do
				id[k] = v
			end
			c:setChild(true)
			id.big_step = nil
			id.small_step = nil
			c:setInternalData(id)
		end
	end
end

local function initialize(mep, _np)
	local d = get_mep_data(mep)
	local np = _np or (d.np or 20)
	mep:resampleStateXYZPath(np)
	
	if not mep:isChild() then
		if d.big_step == nil then
			d.big_step = MEP.new()
			d.small_step = MEP.new()

			d.big_step:setChild(true)
			d.small_step:setChild(true)
		end
	end
	d.isInitialized = true
end

local function single_compute_step(mep, get_site_ss, set_site_ss, get_energy_ss, np)
	local d = get_mep_data(mep)

	if d.isInitialized == nil then
		initialize(mep)
	end

	local movement = 0
	local maxmovement = 0
	mep:calculateEnergyGradients(get_site_ss, set_site_ss, get_energy_ss)
	mep:makeForcePerpendicularToPath(get_site_ss, set_site_ss, get_energy_ss)
	mep:makeForcePerpendicularToSpins(get_site_ss, set_site_ss, get_energy_ss)
	movement, maxmovement = mep:applyForces()
	mep:resampleStateXYZPath(np)

	return movement,maxmovement
end

local function compute(mep, n)
	local d = get_mep_data(mep)
	local ss = getSpinSystem(mep)
	local np = d.np or 20
	local energy_function = d.energy_function
	local tol = getTolerance(mep)
	
	if ss == nil then
		error("SpinSystem is nil. Set a working SpinSystem with :setSpinSystem")
	end
	
	if energy_function == nil then
		error("Energy function is nil. Set an energy function with :setEnergyFunction")
	end
	
	local function get_site_ss(x,y,z)
		local sx,sy,sz = ss:spin(x,y,z)
		return sx,sy,sz
	end
	local function set_site_ss(x,y,z,  sx,sy,sz)
		local ox,oy,oz,m = ss:spin({x,y,z})
		ss:setSpin({x,y,z}, {sx,sy,sz}, m)
	end
	local function get_energy_ss()
		local e = energy_function(ss)
		return e
	end
	
	if d.isInitialized == nil then
		initialize(mep)
	end

	local successful_steps = 0
	n = n or 50
	while successful_steps < n do
		local current_beta = mep:beta()

		copy_to_children(mep)

		d.big_step:setBeta(current_beta)
		local _, maxMovement = single_compute_step(d.big_step, get_site_ss, set_site_ss, get_energy_ss, np)

		d.small_step:setBeta(current_beta/2)
		
		single_compute_step(d.small_step, get_site_ss, set_site_ss, get_energy_ss, np)
		single_compute_step(d.small_step, get_site_ss, set_site_ss, get_energy_ss, np)


		local aDiff, maxDiff = d.big_step:absoluteDifference(d.small_step)
		local aDiffAvrg = aDiff / np
		
		local step_mod, good_step = getStepMod(tol, maxDiff)
		
		if good_step then
			d.small_step:internalCopyTo(mep)
			successful_steps = successful_steps + 1
		end
		mep:setBeta(step_mod * current_beta)
	end
end

local function setSpinSystem(mep, ss)
	local d = get_mep_data(mep)
	if getmetatable(ss) ~= SpinSystem.metatable() then
		error("setSpinSystem requires a SpinSystem", 2)
	end
	d.ss = ss
end


local function setNumberOfPathPoints(mep, n)
	local d = get_mep_data(mep)
	if type(n) ~= "number" then
		error("setNumberOfPathPoints requires a number", 2)
	end
	if n < 2 then
		error("Number of points must be 2 or greater.")
	end
	d.np = n
end

local function setGradientMaxMotion(mep, dt)
	local d = get_mep_data(mep)
	if type(dt) ~= "number" then
		error("setGradientMaxMotion requires a number", 2)
	end
	d.gdt = dt * mep:problemScale()
end


local function gradientMaxMotion(mep)
	local d = get_mep_data(mep)
	return (d.gdt / mep:problemScale()) or 0.1
end


local function setEnergyFunction(mep, func)
	local d = get_mep_data(mep)
	if type(func) ~= "function" then
		error("setEnergyFunction requires a function", 2)
	end
	d.energy_function = func
end

local function setSites(mep, tt)
	if type(tt) ~= "table" then
		error("setSites requires a table of 1,2 or 3 Component Tables representing sites.") 
	end
	
	if tt[1] and type(tt[1]) ~= "table" then
		error("setSites requires a table of 1,2 or 3 Component Tables representing sites.") 
	end
	
	mep:clearSites()
	for k,v in pairs(tt) do
		mep:_addSite(v[1], v[2], v[3])
	end
end

local function setInitialPath(mep, pp)
	local msg = "setInitialPath requires a Table of Tables of site orientations"
	local tableType = type({})
	if type(pp) ~= tableType then
		error(msg)
	end
	local numSites = nil
	mep:clearPath()
	for p=1,table.maxn(pp) do
		if type(pp[p]) ~= tableType then
			error(msg)
		end
		numSites = numSites or table.maxn(pp[p])
		if numSites ~= table.maxn(pp[p]) then
			error("Site count mismatch at path point number " .. p)
		end
		
		for s=1,numSites do
			if type(pp[p][s]) ~= tableType then
				error(msg)
			end
			local x = pp[p][s][1] or 0
			local y = pp[p][s][2] or 0
			local z = pp[p][s][3] or 0
			mep:_addStateXYZ(x,y,z)
		end
	end
-- 	print("PP=", table.maxn(pp))
end


local function getNumberOfPathPoints(mep)
	local d = get_mep_data(mep)
	return (d.np or 20)
end

local function getNumberSites(mep)
	return table.maxn(mep:sites())
end

local function isChild(mep)
	local d = get_mep_data(mep)
	return (d.child or false)
end

local function setChild(mep, flag)
	local d = get_mep_data(mep)
	d.child = flag
end


local function relaxSinglePoint(mep, pointNum, numSteps)
	local d = get_mep_data(mep)
	if d.isInitialized == nil then
		initialize(mep)
	end
	local tol = getTolerance(mep)
	local ss = getSpinSystem(mep)
	local energy_function = getEnergyFunction(mep)

	local function get_site_ss(x,y,z)
		local sx,sy,sz = ss:spin(x,y,z)
		return sx,sy,sz
	end
	local function set_site_ss(x,y,z,  sx,sy,sz)
		local _,_,_,m = ss:spin({x,y,z})
		ss:setSpin({x,y,z}, {sx,sy,sz}, m)
	end

	local function get_energy_ss()
		local e = energy_function(ss)
		return e
	end

	local n = numSteps or 50
	local completed_steps = 0
	while completed_steps < n do
		local current_beta = mep:beta()

		copy_to_children(mep)

		d.big_step:setBeta(current_beta)
		d.big_step:_relaxSinglePoint(pointNum, get_site_ss, set_site_ss, get_energy_ss, 1)
		
		d.small_step:setBeta(current_beta/2)
		local _, m1 = d.small_step:_relaxSinglePoint(pointNum, get_site_ss, set_site_ss, get_energy_ss, 2)
		
		local aDiff, maxDiff = d.big_step:absoluteDifference(d.small_step, pointNum)
-- 		local aDiffAvrg = aDiff / np
			
		local step_mod, good_step = getStepMod(tol, maxDiff, m1)
		
		if good_step then
			d.small_step:internalCopyTo(mep)
			completed_steps = completed_steps + 1
		end
		mep:setBeta(step_mod * current_beta)
		
	end
end



-- compute hessian, get eigenvector of negative curvature
-- negate that direction in the gradient and step
local function relaxSaddlePoint(mep, pointNum, numSteps)
	local d = get_mep_data(mep)
-- 	if d.isInitialized == nil then
-- 		initialize(mep)
-- 	end
	local ss = getSpinSystem(mep)
	local energy_function = getEnergyFunction(mep)
	local tol = getTolerance(mep)

	local function get_site_ss(x,y,z)
		local sx,sy,sz = ss:spin(x,y,z)
		return sx,sy,sz
	end
	local function set_site_ss(x,y,z,  sx,sy,sz)
		local _,_,_,m = ss:spin({x,y,z})
		ss:setSpin({x,y,z}, {sx,sy,sz}, m)
	end

	local function get_energy_ss()
		local e = energy_function(ss)
		return e
	end
	
	numSteps = numSteps or 50
	local completed_steps = 0
	while completed_steps < numSteps do
		local current_beta = mep:beta()
		local D = mep:hessianAtPoint(pointNum)
		local vals, vecs = D:matEigen()
		
		local minv, mini = vals:min()
		
		local scaled_vec = {}
		for i=1,vecs:nx() do
			local ee = vecs:slice({{1,i}, {vecs:nx(),i}})
			ee:scale(vals:get(i))
			table.insert(scaled_vec, ee)
		end
		
		for i=2,vecs:nx() do
			scaled_vec[1]:pairwiseScaleAdd(1, scaled_vec[i], scaled_vec[1])
		end
		
		local down_dir = scaled_vec[1] --vecs:slice({{1,mini}, {vecs:nx(),mini}}):toTable(1)
		
		down_dir:scale(-1/ (down_dir:dot(down_dir)^(1/2)))
		
		down_dir = down_dir:toTable(1)

		
		local down_dir = vecs:slice({{1,mini}, {vecs:nx(),mini}}):toTable(1)

		
		copy_to_children(mep)

		d.big_step:setBeta(current_beta)
		d.big_step:_relaxSinglePoint(pointNum, get_site_ss, set_site_ss, get_energy_ss, 1, down_dir)


		d.small_step:setBeta(current_beta/2)
		local _, maxMovement = d.small_step:_relaxSinglePoint(pointNum, get_site_ss, set_site_ss, get_energy_ss, 2, down_dir)

		local _, maxDiff = d.big_step:absoluteDifference(d.small_step, pointNum)
		
		local step_mod, good_step = getStepMod(tol, maxDiff, maxMovement)
		
		if good_step then
			d.small_step:internalCopyTo(mep)
			completed_steps = completed_steps + 1
		end
		mep:setBeta(step_mod * current_beta)
-- 		completed_steps = completed_steps + 1

-- 		print(mep:beta(), good_step)
		
		
-- 		mep:_relaxSinglePoint(pointNum, get_site_ss, set_site_ss, get_energy_ss, stepSize, 1, down_dir)
	end
end


local function hessianAtPoint(mep, pointNum, destArray)
	local d = get_mep_data(mep)
	local ss = getSpinSystem(mep)
	local energy_function = getEnergyFunction(mep)
	
	local function get_site_ss(x,y,z)
		local sx,sy,sz = ss:spin(x,y,z)
		return sx,sy,sz
	end
	local function set_site_ss(x,y,z,  sx,sy,sz)
		local _,_,_,m = ss:spin({x,y,z})
		ss:setSpin({x,y,z}, {sx,sy,sz}, m)
	end
	local function get_energy_ss()
		local e = energy_function(ss)
		return e
	end

	
	local c = mep:siteCount()
	
	if destArray then
		if destArray:nx() ~= c*3 or destArray:ny() ~= c*3 then
			error("Destination array size mismatch. Expected " .. c*3 .. "x" .. c*3)
		end
	else
		destArray = Array.Double.new(c*3, c*3)
	end

	local t = mep:_hessianAtPoint(pointNum, get_site_ss, set_site_ss, get_energy_ss)

	for x=0,c*3-1 do
		for y=0,c*3-1 do
			destArray:set(x+1, y+1, t[x+y*(c*3)+1])
		end
	end
			
	return destArray
end

local function maximalPoints(mep)
	pathEnergy(mep) -- calculate energies
	return mep:_maximalPoints()
end

local function energyBarrier(mep)
	local mins, maxs, all = mep:maximalPoints()
	
	local ee = mep:pathEnergy()
	
	local start_e = ee[1]
	
	local max_e = start_e
	
	for k,p in pairs(maxs) do
		if ee[p] > max_e then
			max_e = ee[p]
		end
	end
	
	return max_e - start_e
end

local function sub_mep(mep, p1, p2)
	if p2 < p1 then
		return nil
	end
	if p1 == p2 then
		return nil
	end
	local smep = MEP.new()
	local id ={}
	for k,v in pairs(get_mep_data(mep)) do
		id[k] = v
	end
	smep:setInternalData(id)
		
	local ns = mep:numberOfSites()
	local sub_path = {}
	for p=p1,p2 do
		local state = {}
		for s=1,ns do
			local x, y, z = mep:spin(p, s)
			table.insert(state, {x,y,z})
		end
		table.insert(sub_path, state)
	end
	
	smep:setNumberOfPathPoints(p2-p1+1)
	smep:setSites(mep:sites())
	smep:setInitialPath(sub_path)

	return smep
end


local function splitAtPoint(mep, ...)
	local arg = {...}
	local ret_style
	
	if type(arg[1]) == "table" then
		ret_style = "t"
		local t = {}
		for k,v in pairs(arg[1]) do
			t[k] = v
		end
		arg = t
	else
		ret_style = "l"
	end
	
	table.sort(arg)
	table.insert(arg, mep:numberOfPathPoints())
	table.insert(arg, 1, 1)

	local meps = {}
	
	for i=1,table.maxn(arg)-1 do
		table.insert(meps, sub_mep(mep, arg[i], arg[i+1]))
	end
	
	if ret_style == "t" then
		return meps
	end
	
	local function variadic_return(t)
		local first = t[1]
		table.remove(t, 1)
		
		if table.maxn(t) == 0 then
			return first
		end
		return first, variadic_return(t)
	end
	
	variadic_return(meps)
end



mt.setEnergyFunction = setEnergyFunction
mt.setSites = setSites
mt.setNumberOfPathPoints = setNumberOfPathPoints

mt.energyFunction = getEnergyFunction
mt.numberOfPathPoints = getNumberOfPathPoints
mt.numberOfSites = getNumberSites

mt.compute = compute
mt.pathEnergy = pathEnergy
mt.randomize = randomize

mt.writePathPointTo = writePathPointTo
mt.initialize = initialize

mt.setInitialPath = setInitialPath

mt.setSpinSystem = setSpinSystem

mt.relaxSinglePoint = relaxSinglePoint
mt.relaxSaddlePoint = relaxSaddlePoint

mt.hessianAtPoint = hessianAtPoint

mt.isChild = isChild
mt.setChild = setChild

mt.setGradientMaxMotion = setGradientMaxMotion
mt.gradientMaxMotion = gradientMaxMotion

mt.energyBarrier = energyBarrier

mt.maximalPoints = maximalPoints

mt.splitAtPoint = splitAtPoint

mt.tolerance = getTolerance
mt.setTolerance = setTolerance

MODTAB.help =
function(x)
	if x == setSpinSystem then
		return "Set the SpinSystem that will be used to do energy and orientation calculations. The any changes made to this SpinSystem will be undone before control is returned to the calling environment.",
				"1 *SpinSystem*: SpinSystem to be used in calculations",
				""
	end
	if x == splitAtPoint then
		return "Split the path at the given point(s).",
				"N Integers or a Table of Integers: The location(s) of the split(s). Split points at the ends will be ignored.",
				"N+1 Minimum Energy Pathway Objects or a table of Minimum Energy Pathway Objects: If a table was given as input then a table will be given as output. If 1 or more integers was given as input then 2 or more objects will be returned. Example: <pre>a,b,c = mep:splitAtPoint(12, 17)\n</pre> or <pre> meps = mep:splitAtPoint({12, 17})\n</pre>"
	end
	if x == maximalPoints then
		return "Get the path points that represent local minimums and maximums along the energy path.",
				"",
				"3 Tables: List of path points representing minimums, list of path points representing maximums and combined, sorted list."
	end
	if x == energyBarrier then
		return "Calculate the difference in energy between the initial point and the maximum energy",
				"",
				"1 Number: The energy barrier"
	end
	if x == writePathPointTo then
		return 
			"Write path point to the given spin system",
			"1 Integer, 1 *SpinSystem*: The integer ranges from 1 to the number of path points, the *SpinSystem* will have the sites involved in the Minimum Energy Pathwaycalculation changed to match those at the given path index.",
			""
	end
	if x == initialize then
		return 
			"Expand the endpoints into a coherent rotation over the number of path points specified with setNumberOfPathPoints.",
			"1 Optional integer: A non-default number of points to interpolate over.",
			""
	end
	if x == compute then
		return
			"Run several relaxation steps of the Minimum Energy Pathway method",
			"1 Optional Integer: Number of steps, default 50",
			"1 Number: Absolute amount of path movement in the final step divided by the number of points."
	end
	if x == setEnergyFunction then
		return
			"Set the function used to determine system energy for the calculation.",
			"1 Function: energy calculation function, expected to be passed a *SpinSystem*.",
			""
	end
	if x == setSites then
		return
			"Set the sites that are allowed to move to transition from the initial configuration to the final configuration.",
			"1 Table of 1,2 or 3 Component Tables: mobile sites.",
			""
	end
	if x == setNumberOfPathPoints then
		return
			"Set the number of path points used to approximate a line (defualt 20).",
			"1 Number: Number of path points",
			""
	end
	if x == randomize then
		return
			"Perturb all points (except endpoints) by a random value.",
			"1 Number: Magnitude of perturbation. This value will be scaled by the problemScale().",
			""
	end
	
	
	if x == setInitialPath then
		return
			"Set the initial path for the Minimum Energy Pathway calculation",
			"1 Table of Tables of site orientations: Must be at least 2 elements long to define the start and end points. Example:\n<pre>upup     = {{0,0,1},{0,0,1}}\ndowndown = {{0,0,-1},{0,0,-1}}\n mep:setInitialPath({upup,downdown})\n</pre>",
			""
	end
	if x == getEnergyFunction then
		return
			"Get the function used to determine system energy for the calculation.",
			"",
			"1 Function: energy calculation function, expected to be passed a *SpinSystem*."
	end
	if x == getNumberOfPathPoints then
		return
			"Get the number of path points used to approximate a line (defualt 20).",
			"",
			"1 Integer: Number of path points"
	end
	if x == getNumberSites then
		return
			"Get the number of sites used in calculation.",
			"",
			"1 Integer: Number of sites."
	end
	
	if x == relaxSinglePoint then
		return 	"Allow a single point to move along the local energy gradient either down to a minimum or up to a maximum",
				"1 Integer, 1 Number, 1 Integer: Point to relax,, number of iterations.",
				""
	end

		
	if x == relaxSaddlePoint then
		return 	"Allow a single point to move along the local energy gradient either down to a minimum or up to a maximum. A single gradient coordinate will be inverted based on the 2nd derivative to converge to a saddle point.",
				"1 Integer, 1 Integer: Point to relax, number of iterations.",
				""
	end

	if x == hessianAtPoint then
		return "Compute the 2nd order partial derivative at a given point along the path.",
				"1 Integer, 1 Number, 1 Optional Array: Point to calculate 2nd derivative about, step size used in numerical differentiation (will be scaled by problemScale()), optional destination array. If no array is provided one will be created",
				"1 Array: Derivatives"
	end
	
	if x == pathEnergy then
		return "Get all energies along the path",
				"",
				"1 Table: energies along the path"
	end
	
	if x == getTolerance then
		return "Get the tolerance used in the adaptive algorithm",
				"",
				"1 Number: Tolerance"
	end
	
	if x == setTolerance then
		return "Set the tolerance used in the adaptive algorithm",
				"1 Number: Tolerance. Usually something on the order of 0.1 should be OK.",
				""
	end
	
	-- calling fallback
	if x == nil then
		return help()
	end
	return help(x)
end
