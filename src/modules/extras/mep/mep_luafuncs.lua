-- MEP
local MODNAME = "MEP"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

-- trying something new for style/readability. 
-- Putting docs and code in a table which will later be used
-- to build metatable and help system.
local methods = {}


-- internal support functions
local function get_mep_data(mep)
	if mep == nil then
		return {}
	end
	if mep:getInternalData() == nil then
		mep:setInternalData({})
	end
	return mep:getInternalData()
end


-- helper function to generate common closures
-- usage:
-- get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
local function build_gse_closures(mep)
	local d = get_mep_data(mep)
	local ss = mep:spinSystem()
	local energy_function = mep:energyFunction()

	local function get_site_ss(x,y,z)
		local sx,sy,sz = ss:spin(x,y,z)
		return sx,sy,sz
	end
	local function set_site_ss(x,y,z,sx,sy,sz) --return if a change was made
		local ox,oy,oz,m = ss:spin({x,y,z})
		ss:setSpin({x,y,z}, {sx,sy,sz}, m)
		return (ox ~= sx) or (oy ~= sy) or (oz ~= sz)
	end
	local function get_energy_ss()
		local e = energy_function(ss)
		return e
	end
	return get_site_ss, set_site_ss, get_energy_ss
end


local function getStepMod(tol, err, maxMotion)
	-- print("getStepMod(tol=" .. tol .. ",err=" .. err .. ")")
	if maxMotion then
		if maxMotion > 0.1 then
			-- print("ret 0.5, false")
			return 0.5, false
		end
	end
	if err == 0 then
		-- print("ret 2, true")
		return 2, true
	end

	if err<tol then
		local x = 0.9*(tol / err)^(0.5)
-- 		print(x)
		-- print(err, tol)
		-- print("ret " .. x .. ", " .. tostring(err<tol))
		return x, err<tol
	end
	
	-- print("ret " .. 0.95 * (tol / err)^(0.9) .. ", " .. tostring(err<tol))
	return 0.95 * (tol / err)^(0.9), err<tol
end



methods["spinSystem"] =
{
	"Get the *SpinSystem* used in the calculation",
	"",
	"1 SpinSystem",
	function(mep)
		local d = get_mep_data(mep)
		return d.ss
	end
}

methods["tolerance"] =
{
	"Get the tolerance used in the adaptive algorithm",
	"",
	"1 Number: Tolerance",
	function(mep)
		local d = get_mep_data(mep)
		return d.tol or 1e-4
	end
}

methods["setTolerance"] =
{
	"Set the tolerance used in the adaptive algorithm",
	"1 Number: Tolerance. Usually something on the order of 0.1 should be OK. If tolerance is less than or equal to zero then no adaptive steps will be attempted.",
	"",
	function(mep, tol)
		local d = get_mep_data(mep)
		d.tol = tol
	end
}

methods["energyFunction"] =
{
	"Get the function used to determine system energy for the calculation.",
	"",
	"1 Function: energy calculation function, expected to be passed a *SpinSystem*.",
	function(mep)
		local d = get_mep_data(mep)
		return (d.energy_function or function() return 0 end)
	end
}



-- write a state to a spinsystem
methods["writePathPointTo"] =
{
	"Write path point to the given spin system",
	"1 Integer, 1 *SpinSystem*: The integer ranges from 1 to the number of path points, the *SpinSystem* will have the sites involved in the Minimum Energy Pathwaycalculation changed to match those at the given path index.",
	"",
	function(mep, path_point, ssNew)
		local d = get_mep_data(mep)
		local ss = d.ss
		
		ss:copySpinsTo(ssNew)
		-- 	print("pp", path_point)
		local sites = mep:sites()
		for i=1,table.maxn(sites) do
			local x,y,z,m = mep:spin(path_point, i)
			-- 		print(table.concat(sites[i], ","))
			ssNew:setSpin(sites[i], {x,y,z}, m)
		end
	end
}

methods["energyAtPoint"] =
{
	"Get the energy at a point",
	"1 Integer: Point number",
	"1 Number: Energy at point.",
	function(mep, p)
		local path = mep:pathEnergy()
		return path[p]
	end
}

methods["pathEnergy"] =
{
	"Get all energies along the path",
	"",
	"1 Table: energies along the path",
	function(mep)
		get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
		
		mep:calculateEnergies(get_site_ss, set_site_ss, get_energy_ss);
		return mep:getPathEnergy()
	end
}



methods["randomize"] = 
{
	"Perturb all points (except endpoints) by a random value.",
	"1 Number: Magnitude of perturbation.",
	"",
	function(mep, magnitude)
		mep:_randomize(magnitude)
	end
}

local function copy_to_children(mep, do_big, do_small)
	do_big = do_big or true
	do_small = do_small or true
	local d = get_mep_data(mep)
	if not mep:isChild() then
		local children = {}
		if do_big then
			mep:internalCopyTo(d.big_step)
			table.insert(children, d.big_step)
		end
		if do_small then
			mep:internalCopyTo(d.small_step)
			table.insert(children, d.small_step)
		end

		for _,c in pairs(children) do
			if c then
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
end


methods["initialize"] = 
{
	"Expand the endpoints into a coherent rotation over the number of path points specified with setNumberOfPathPoints.",
	"1 Optional integer: A non-default number of points to interpolate over.",
	"",
	function(mep, _np)
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
}

local function single_compute_step(mep, get_site_ss, set_site_ss, get_energy_ss, np)
	local d = get_mep_data(mep)

	if d.isInitialized == nil then
		mep:initialize()
	end

	mep:calculateEnergyGradients(get_site_ss, set_site_ss, get_energy_ss)
	mep:makeForcePerpendicularToPath(get_site_ss, set_site_ss, get_energy_ss)
	mep:makeForcePerpendicularToSpins(get_site_ss, set_site_ss, get_energy_ss)
	mep:applyForces()
end


methods["compute"] =
{
	"Run several relaxation steps of the Minimum Energy Pathway method",
	"1 Optional Integer, 1 Optional Number: Number of steps, default 50. Tolerance different than tolerance specified.",
	"1 Number: Number of successful steps taken, for tol > 0 this may be less than number of steps requested",
	function(mep, n, tol)
		local d = get_mep_data(mep)
		local ss = mep:spinSystem()
		local np = d.np or 20
		local energy_function = d.energy_function
		tol = tol or mep:tolerance()
		
		if ss == nil then
			error("SpinSystem is nil. Set a working SpinSystem with :setSpinSystem")
		end
		
		if energy_function == nil then
			error("Energy function is nil. Set an energy function with :setEnergyFunction")
		end
		
        get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
		
		if d.isInitialized == nil then
			mep:initialize()
		end
		
		local successful_steps = 0
		n = n or 50
		for i = 1,n do
			-- 	while successful_steps < n do
			-- 		print(successful_steps)
			local current_beta = mep:beta()
			
			-- first bool below is for "do_big" second is for "do_small"
			copy_to_children(mep, true, tol>=0)
			
			d.big_step:setBeta(current_beta)
			single_compute_step(d.big_step, get_site_ss, set_site_ss, get_energy_ss, np)
			
			if tol > 0 then -- negative tolerance will mean no adaptive steps
				d.small_step:setBeta(current_beta/2)
				
				single_compute_step(d.small_step, get_site_ss, set_site_ss, get_energy_ss, np)
				single_compute_step(d.small_step, get_site_ss, set_site_ss, get_energy_ss, np)
				
				local aDiff, maxDiff, max_idx = d.big_step:absoluteDifference(d.small_step)
				local aDiffAvrg = aDiff / np
				
				-- print("beta = ", current_beta)
				local step_mod, good_step = getStepMod(tol, maxDiff)
				-- 			print(maxDiff, tol)
				if good_step then
					d.small_step:internalCopyTo(mep)
					mep:resampleStateXYZPath(np)
					successful_steps = successful_steps + 1
				end
				mep:setBeta(step_mod * current_beta)
			else -- negative or zero tolerance: no adaptive step
				successful_steps = successful_steps + 1
				d.big_step:internalCopyTo(mep)
			end
			
			mep:resampleStateXYZPath(np)
		end
		
		return successful_steps
	end
}

methods["setSpinSystem"] =
{
	"Set the SpinSystem that will be used to do energy and orientation calculations. The any changes made to this SpinSystem will be undone before control is returned to the calling environment.",
	"1 *SpinSystem*: SpinSystem to be used in calculations",
	"",
	function(mep, ss)
		local d = get_mep_data(mep)
		if getmetatable(ss) ~= SpinSystem.metatable() then
			error("setSpinSystem requires a SpinSystem", 2)
		end
		d.ss = ss
	end
}


methods["setNumberOfPathPoints"] =
{
	"Set the number of path points used to approximate a line (defualt 20).",
	"1 Number: Number of path points",
	"",
	function(mep, n)
		local d = get_mep_data(mep)
		if type(n) ~= "number" then
			error("setNumberOfPathPoints requires a number", 2)
		end
		if n < 2 then
			error("Number of points must be 2 or greater.")
		end
		d.np = n
	end
}

-- local function setGradientMaxMotion(mep, dt)
-- 	local d = get_mep_data(mep)
-- 	if type(dt) ~= "number" then
-- 		error("setGradientMaxMotion requires a number", 2)
-- 	end
-- 	d.gdt = dt * mep:problemScale()
-- end
-- 
-- 
-- local function gradientMaxMotion(mep)
-- 	local d = get_mep_data(mep)
-- 	return (d.gdt / mep:problemScale()) or 0.1
-- end



methods["setEnergyFunction"] =
{
	"Set the function used to determine system energy for the calculation.",
	"1 Function: energy calculation function, expected to be passed a *SpinSystem*.",
	"",
	function(mep, func)
		local d = get_mep_data(mep)
		if type(func) ~= "function" then
			error("setEnergyFunction requires a function", 2)
		end
		d.energy_function = func
	end
}


methods["setSites"] = 
{
	"Set the sites that are allowed to move to transition from the initial configuration to the final configuration.",
	"1 Table of 1,2 or 3 Component Tables: mobile sites.",
	"",
	function(mep, tt)
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
}


methods["site"] = 
{
	"Get a single site.",
	"1 Integer: Site index, positive will count from start, negative will count from end",
	"1 Table of 3 Integers: Site",
	function(mep, idx)
		idx = idx or error("require index")
		if idx < 0 then
			idx = mep:numberOfSites() + idx
		end

		return mep:sites()[idx] or "Invalid index"
	end
}

methods["setInitialPath"] =
{
	"Set the initial path for the Minimum Energy Pathway calculation. Must be called after :setSpinSystem",
	"1 Table of Tables of site orientations or nils: Must be at least 2 elements long to define the start and end points. Example:\n<pre>upupup     = {{0,0,1}, {0,0,1}, {0,0,1}}\ndowndowndc = {{0,0,-1},{0,0,-1},nil}\n mep:setInitialPath({upupup,downdowndc})\n</pre>Values of nil for orientations in the start or end points mean that the algorithm will not attempt to keep them stationary - they will be allowed to drift. Their initial value will be whatever they are in the SpinSystem at the time the method is called. These drifting endpoints are sometimes referred to as `don't care' sites.",
	"",
	function(mep, pp)
		local ss = mep:spinSystem() or error("SpinSystem must be set before :setInitialPath()")
		
		local msg = "setInitialPath requires a Table of Tables of site orientations"
		local tableType = type({})
		if type(pp) ~= tableType then
			error(msg)
		end
		local numSites = mep:numberOfSites()
		local sites = mep:sites()
		
		mep:clearPath()
		for p=1,table.maxn(pp) do
			local mobility = 1
			if p == 1 or p == table.maxn(pp) then
				mobility = 0 --fixed endpoints
			end
			
			if type(pp[p]) ~= tableType then
				error(msg)
			end
			
			for s=1,numSites do
				local x, y, z = ss:spin( sites[s] )
				if pp[p][s] == nil then --user doesn't care
					mep:_setImageSiteMobility(p, s, 1)
				else
					x = pp[p][s][1] or x
					y = pp[p][s][2] or y
					z = pp[p][s][3] or z
					mep:_setImageSiteMobility(p, s, mobility)
				end
				mep:_addStateXYZ(x,y,z)
			end
		end
		-- 	print("PP=", table.maxn(pp))
	end
}



methods["numberOfPathPoints"] =
{
	"Get the number of path points used to approximate a line (defualt 20).",
	"",
	"1 Integer: Number of path points",
	function(mep)
		local d = get_mep_data(mep)
		return (d.np or 20)
	end
}

methods["numberOfSites"] = 
{
	"Get the number of sites used in calculation.",
	"",
	"1 Integer: Number of sites.",
	function(mep)
		return table.maxn(mep:sites())
	end
}

methods["isChild"] =
{
	nil,nil,nil,
	function(mep)
		local d = get_mep_data(mep)
		return (d.child or false)
	end
}

methods["setChild"] = 
{
	nil,nil,nil,
	function(mep, flag)
		local d = get_mep_data(mep)
		d.child = flag
	end
}

methods["pathEnergyNDeriv"] = 
{
	"Calculate the Nth derivative of the path energy.",
	"1 Integer: The derivative, 0 = energy, 1 = slope of energy, 2 = acceleration of energy, ...",
	"3 Tables: The Nth derivatives of the energy, the normalized location along the path between 0 and 1, the list of zero crossings",
	function(mep, n)
		local d = get_mep_data(mep)
		if d.isInitialized == nil then
			mep:initialize()
		end
		
        get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
		
		return mep:_pathEnergyNDeriv(n, get_site_ss, set_site_ss, get_energy_ss)
	end
}

local function relaxPoint(mep, pointNum, steps, goal)
	local mep_data = get_mep_data(mep)
	if mep_data.isInitialized == nil then
		mep:initialize()
	end
	local energy_function = mep:energyFunction()
	local tol = nonDefaultTol or mep:tolerance()
		
	get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)

	steps = steps or 10
	goal = goal or 0
	local h, good_steps = nil
	for i=1,steps do
		min_mag_grad, h, good_steps = mep:_relaxSinglePoint_sd(pointNum, get_site_ss, set_site_ss, get_energy_ss, h, goal)
		if min_mag_grad < goal then
			return min_mag_grad
		end
	end

	-- print(string.format("%8e %8e %d", min_mag_grad, h, good_steps))
	return min_mag_grad
end

methods["relaxSinglePoint"] =
{
	"Move point to minimize energy gradient. This will find local maxs, mins or saddle points.",
	"1 Integer, 1 Optional Integer, 1 Optional Number, 1 Optional Number: Point to relax, optional number of iterations (default 50), optional goal gradient magnitude for early completion, stall value: quit if steps are smaller than this value.",
	"2 Numbers: Gradient magnitude at final iteration, number of steps taken",
	function(mep, pointNum, numSteps, goal)
		return relaxPoint(mep, pointNum, numSteps, goal)
	end
}



methods["slidePoint"] =
{
	"Move point along energy gradient",
	"1 Integer, 1 Optional Integer, 1 Optional Number: Point to relax, direction (+1 for increase energy, -1 for decrease energy). optional number of iterations (default 50), optional step size (default epsilon) this step size will be adapted.",
	"1 Number: Final step size.",
	function(mep, pointNum, direction, numSteps, stepSize) --, nonDefaultTol)
		local d = get_mep_data(mep)
		if d.isInitialized == nil then
			mep:initialize()
		end
		
        get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
		
		local numSteps = numSteps or 50

		if pointNum == nil then
			error("Require point number")
		end

		if direction == nil then
			direction = -1
		end

		return mep:_slidePoint(get_site_ss, set_site_ss, get_energy_ss, pointNum, direction, numSteps, stepSize)
	end
}



methods["classifyPoint"] =
{
	"Determine if a point is a local maximum, minimum or neither",
	"1 Integer, 1 optional Number: Point index, optional step size",
	"1 String: \"Maximum\", \"Minimum\" or \"\"",
	function(mep, pointNum, h)
		local d = get_mep_data(mep)
		if d.isInitialized == nil then
			mep:initialize()
		end
		
        get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
 		
		return mep:_classifyPoint(get_site_ss, set_site_ss, get_energy_ss, pointNum, h)
	end
}

methods["relaxSaddlePoint"] = 
{
	"Obsolete. This function calls relaxSinglePoint.",
	"",
	"",
	function(mep, pointNum, numSteps, nonDefaultTol)
		relaxPoint(mep, pointNum, numSteps)
	end	
}

methods["flatAtPoint"] =
{
	"Determine if a point is flat. This is done by comparing the magnitude of the gradient to a small number which may be included (default 1e-8).",
	"1 Integer, 1 Optional Number: Point to calculate gradient about, number to compare the magnitude to to determine flattness.",
	"1 Boolean: True if flat, False otherwise.",
	function(mep, pointNum, small_number)
		small_number = small_numer or 1e-8
		local grad = mep:gradientAtPoint(pointNum)
		return (grad:dot(grad))^(1/2) < small_number
	end
}

methods["gradientAtPoint"] =
{
	"Compute the gradient at a given point.",
	"1 Integer, 1 Optional Array: Point to calculate gradient about. If no array is provided one will be created",
	"1 Array: Gradient",
	function(mep, pointNum, destArray)
		local d = get_mep_data(mep)
		local ss = mep:spinSystem()
		local energy_function = mep:energyFunction()
		local c = mep:siteCount()
		
		if destArray then
			if destArray:nx() ~= c*3 or destArray:ny() ~= 1 then
				error("Destination array size mismatch. Expected " .. c*3 .. "x" .. 1)
			end
		else
			destArray = Array.Double.new(c*3, 1)
		end
		
		local t = mep:_gradAtPoint(pointNum, get_site_ss, set_site_ss, get_energy_ss)
		
		for x=1,c*3 do
			destArray:set(x, 1, t[x])
		end
		
		return destArray
	end
}

methods["secondDerivativeAtPoint"] = {
	"Compute d2 E / dC1 dC2 at a point",
	"1 Integer, 2 Integers, 2 optional Numbers: The first integer is the point number. The next 2 integers are coordinate numbers. Valid values are between 1 and then number of sites time 3. A 2 Site problem would accept numbers between 1 and 6 inclusively. The optional numbers are step sizes for the numeric differentiaion",
	"1 Number: second derivative of energy with repect to the coordinates",
	function(mep, pointNum, c1, c2, h1, h2)
        get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)

		if c1 < 1 or c1 > mep:numberOfSites()*3 then
			error("c1 out of bounds")
		end

		if c2 < 1 or c2 > mep:numberOfSites()*3 then
			error("c2 out of bounds")
		end

		if pointNum < 1 or pointNum > mep:numberOfPathPoints() then
			error("point index out of bounds")
		end

		local t = mep:_computePointSecondDerivativeAB(pointNum, c1, c2, get_site_ss, set_site_ss, get_energy_ss, h1, h2)
		return t
	end

}

methods["hessianAtPoint"] = {
	"Compute the 2nd order partial derivative at a given point along the path.",
	"1 Integer, 1 Optional Array: Point to calculate 2nd derivative about. If no array is provided one will be created",
	"1 Array: Derivatives",
	function(mep, pointNum, destArray)
        get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
		
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
}

methods["coordinateComponentNames"] = 
{
	"Get the short and long names for each coordinate component given a coordinate type",
	"1 Optional String: \"Cartesian\", \"Canonical\" or \"Spherical\". If no string is provided then the current coordinate system is assumed.",
	"2 Tables of Strings: Short and long forms",
	function(mep, s)
		s = s or mep:coordinateSystem()
		local data = {}
		data["cartesian"] = {{"x", "y", "z"}, {"X Axis", "Y Axis", "Z Axis"}}
		data["spherical"] = {{"r", "p", "t"}, {"Radial", "Azimuthal", "Polar"}}
		data["canonical"] = {{"r", "phi", "p"},{"Radial", "Azimuthal", "cos(Polar)"}}
		if s == nil then
			error("method requires a coordinate system name")
		end
		if data[string.lower(s)] == nil then
			error("method requires a validcoordinate system name")
		end
		local t = data[string.lower(s)]
		return t[1], t[2]
	end
	
}

methods["maximalPoints"] = 
{
	"Obsolete: calls criticalPoints",
	"",
	"",
	function(mep)
		mep:pathEnergy() -- calculate energies
		return mep:_maximalPoints()
	end
}

methods["criticalPoints"] = 
{
	"Get the path points that represent local minimums and maximums along the energy path.",
	"",
	"3 Tables: List of path points representing minimums, list of path points representing maximums and combined, sorted list.",
	function(mep)
		mep:pathEnergy() -- calculate energies
		return mep:_maximalPoints()
	end
}


methods["reduceToPoints"] = 
{
	"Reduce the path to the given set of points. The points between the given points will be discarded and the total number of points will be reduced to the given number of points.",
	"N Integers or 1 Table of integers: The points to keep while discarding the others.",
	"",
	function(mep, ...)
		if type(arg[1]) == type({}) then -- we were given a table
			arg = arg[1]
		end
		local new_initial_path = {}
		local n = 0
		for i,v in pairs(arg) do
			local t = {}
			for j=1,mep:numberOfSites() do
				local x,y,z = mep:spin(v, j)
				t[j] = {x,y,z}
			end
			new_initial_path[i] = t
			n = n + 1
		end

		mep:setInitialPath(new_initial_path)
		mep:setNumberOfPathPoints(n)
	end
}


methods["reduceToCriticalPoints"] = 
{
	"Reduce the path to the points returned from the :criticalPoints() method. The points between the critical points will be discarded and the total number of points will be reduced to the number of critical points.",
	"",
	"",
	function(mep)
		local _, _, cps = mep:criticalPoints()
		mep:reduceToPoints(cps)
	end
}


methods["energyBarrier"] =
{
	"Calculate the difference in energy between the initial point and the maximum energy",
	"",
	"1 Number: The energy barrier",
	function(mep)
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
}


-- internal support function
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


methods["splitAtPoint"] =
{
	"Split the path at the given point(s).",
	"N Integers or a Table of Integers: The location(s) of the split(s). Split points at the ends will be ignored.",
	"N+1 Minimum Energy Pathway Objects or a table of Minimum Energy Pathway Objects: If a table was given as input then a table will be given as output. If 1 or more integers was given as input then 2 or more objects will be returned. Example: <pre>a,b,c = mep:splitAtPoint(12, 17)\n</pre> or <pre> meps = mep:splitAtPoint({12, 17})\n</pre>",
	function(mep, ...)
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
}


methods["merge"] =
{
	"Combine multiple MEPs into a single MEP.",
	"N MEPs or a Table of MEPs: MEPs to merge. It is assumed that final point and initial point in consecutive MEPs coincide.",
	"",
	function(mep, ...)
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

		local function point(mep, p)
			local t = {}
			local s = mep:numberOfSites()
			for i=1,s do
				local x,y,z = mep:spin(p, i) 
				t[i] = {x,y,z}
			end
			return t
		end
		
		local initial_path = {}
		table.insert(initial_path, point(arg[1], 1))
		for k,v in pairs(arg) do
			for i=2,v:numberOfPathPoints() do
				table.insert(initial_path, point(v, i))
			end
		end

		mep:setNumberOfPathPoints(table.maxn(initial_path))
		mep:setInitialPath(initial_path)
		mep:initialize()

	end
}


methods["convertCoordinateSystem"] =
{
	"Convert a vector from a source coordinate system to a destination coordinate system",
	"2 Strings, 1 Table of 3 Numbers or 3 Numbers: Source and Destination Coordinate System names (must match one" ..
		"of the return values from MEP:coordinateSystems() ) and a vector given either as a table of nubmers or as 3 numbers.",
	"1 Table of 3 Numbers or 3 Numbers and 1 String: The return transformed vector will be returned in the same format as the input vector and the name of the new coordinate system will be returned as the last argument",

	function(mep, src_cs, dest_cs, ...)
		local possible_cs = mep:coordinateSystems()
		local src_cs_idx = nil
		local dest_cs_idx = nil
		local input_as_table = nil
		for k,v in pairs(possible_cs) do
			if string.lower(v) == string.lower(src_cs) then
				src_cs_idx = k-1
			end
			if string.lower(v) == string.lower(dest_cs) then
				dest_cs_idx = k-1
			end
		end

		if src_cs_idx == nil then error("Invalid source coordinate system") end
		if dest_cs_idx == nil then error("Invalid destination coordinate system") end

		if type(arg[1]) == type({}) then -- we were given a table
            arg = arg[1]
			input_as_table = true
        end

		local a,b,c = arg[1], arg[2], arg[3]
		a,b,c = mep:_convertCoordinateSystem(a,b,c, src_cs_idx, dest_cs_idx)

		if input_as_table then
			return {a,b,c}, possible_cs[dest_cs_idx+1]
		end
		return a,b,c, possible_cs[dest_cs_idx+1]
	end
}


-- inject above into existing metatable for MEP operator
for k,v in pairs(methods) do
    t[k] = v[4]
end

-- backup old help function for fallback
local help = MODTAB.help

-- create new help function for methods above
MODTAB.help = function(x)
    for k,v in pairs(methods) do
        if x == v[4] then
            return v[1], v[2], v[3]
        end
    end

    -- fallback to old help if a case is not handled
    if x == nil then -- MEP overview
		return [[
Calculates a minimum energy pathway between two states. There are ##return table.maxn(MEP.new():coordinateSystems())## coordinate systems to choose from: Cartesian, Spherical and Canonical.

The Cartesian coordinate system has the standard coordinates along orthogonal axis: x, y, z. 
The gradient is defined as:
$ \nabla = \hat{x} \frac{\partial}{\partial x} + \hat{y} \frac{\partial}{\partial y} + \hat{z} \frac{\partial}{\partial z}$ 

The Spherical coordinate system follows physics conventions with coordinates $r$, $\phi$, $\theta$ as Radial, Azimuthal and Polar directions. 
The gradient is defined as:
$ \nabla = \hat{r} \frac{\partial}{\partial r} + \frac{1}{r} \hat{\theta} \frac{\partial}{\partial \theta} + \frac{1}{r\: sin(\theta)} \hat{\phi} \frac{\partial}{\partial \phi}$ 

The Canonical coordinate system is a modified spherical system with coordinates $r$, $\phi$, $p = cos(\theta)$ as Radial, Azimuthal and Cosine Polar directions. 
The gradient is defined as:
$ \nabla = \hat{r} \frac{\partial}{\partial r} + (r^2 (1-p^2))^{-\frac{1}{2}} \hat{\theta} \frac{\partial}{\partial \theta} + (r^2/(1-p^2))^{-\frac{1}{2}} \hat{p} \frac{\partial}{\partial p}$ 
]], "", ""
    end
    return help(x)
end


