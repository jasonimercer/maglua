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
	mep:_randomize(magnitude * mep:problemScale())
end

local function initialize(mep, _np)
	local d = get_mep_data(mep)
	local np = _np or (d.np or 20)
	mep:resampleStateXYZPath(np)
	
	if _np == nil then
		d.isInitialized = true
	end
end

local function compute(mep, n)
	local d = get_mep_data(mep)
	local ss = d.ss
	local energy_function = d.energy_function
	local np = d.np or 20

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
-- 		print(string.format("changing site {%g %g %g} from {%g %g %g} to {%g %g %g}", x,y,z, ox,oy,oz, sx,sy,sz))
		ss:setSpin({x,y,z}, {sx,sy,sz}, m)
	end

	local function get_energy_ss()
		local e = energy_function(ss)
		return e
	end
	
	if d.isInitialized == nil then
		initialize(mep)
	end

	local movement = 0
	n = n or 50
	local ps =  mep:problemScale()
	local dt = mep:gradientMaxMotion()
	for i=1,n do
		mep:calculateEnergyGradients(get_site_ss, set_site_ss, get_energy_ss)
		mep:makeForcePerpendicularToPath(get_site_ss, set_site_ss, get_energy_ss)
		mep:makeForcePerpendicularToSpins(get_site_ss, set_site_ss, get_energy_ss)
		movement = mep:applyForces()
		mep:resampleStateXYZPath(np)
	end
	
	return movement / np
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

local function getSpinSystem(mep)
	local d = get_mep_data(mep)
	return d.ss
end

local function getEnergyFunction(mep)
	local d = get_mep_data(mep)
	return (d.energy_function or function() return 0 end)
end

local function getNumberOfPathPoints(mep)
	local d = get_mep_data(mep)
	return (d.np or 20)
end

local function getNumberSites(mep)
	return table.maxn(mep:sites())
end


local function relaxSinglePoint(mep, pointNum, stepSize, numSteps)
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

	local ps =  mep:problemScale()
	stepSize = stepSize * ps^2

	-- these or X are for default values
	mep:_relaxSinglePoint(pointNum, get_site_ss, set_site_ss, get_energy_ss, stepSize, numSteps)
end



-- compute hessian, get eigenvector of negative curvature
-- negate that direction in the gradient and step
local function relaxSaddlePoint(mep, pointNum, stepSize, numSteps)
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
	
	local ps =  mep:problemScale()
	stepSize = stepSize * ps^2
	
	for i=1,numSteps do
		local D = mep:hessianAtPoint(pointNum, 1e-8*ps)
		local vals, vecs = D:matEigen()
		
		local minv, mini = vals:min()
		
		local down_dir = vecs:slice({{1,mini}, {vecs:nx(),mini}}):toTable(1)
		
-- 		print(table.concat(down_dir, ", "))
		
		mep:_relaxSinglePoint(pointNum, get_site_ss, set_site_ss, get_energy_ss, stepSize, 1, down_dir)
	end
end


local function hessianAtPoint(mep, pointNum, h, destArray)
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
	
	if h <= 0 then
		error("Numerical derivative step size (h) must be positive")
	end
	
	
	local c = mep:siteCount()
	
	if destArray then
		if destArray:nx() ~= c*3 or destArray:ny() ~= c*3 then
			error("Destination array size mismatch. Expected " .. c*3 .. "x" .. c*3)
		end
	else
		destArray = Array.Double.new(c*3, c*3)
	end

	local t = mep:_hessianAtPoint(pointNum, h, get_site_ss, set_site_ss, get_energy_ss)

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


mt.setGradientMaxMotion = setGradientMaxMotion
mt.gradientMaxMotion = gradientMaxMotion

mt.energyBarrier = energyBarrier

mt.maximalPoints = maximalPoints

mt.splitAtPoint = splitAtPoint

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
				"1 Integer, 1 Number, 1 Integer: Point to relax, step size (positive for down, negative for up), number of iterations.",
				"3 Numbers: Initial Energy, New energy and absolute energy change"
	end

		
	if x == relaxSaddlePoint then
		return 	"Allow a single point to move along the local energy gradient either down to a minimum or up to a maximum. A single gradient coordinate will be inverted based on the 2nd derivative to converge to a saddle point.",
				"1 Integer, 1 Number, 1 Integer: Point to relax, step size (positive for down, negative for up), number of iterations.",
				"3 Numbers: Initial Energy, New energy and absolute energy change"
	end

	if x == hessianAtPoint then
		return "Compute the 2nd order partial derivative at a given point along the path.",
				"1 Integer, 1 Number, 1 Optional Array: Point to calculate 2nd derivative about, step size used in numerical differentiation (will be scaled by problemScale()), optional destination array. If no array is provided one will be created",
				"1 Array: Derivatives"
	end
		
	
	if x == setGradientMaxMotion then
		return "Set the max avrg motion relative to moment strength used in the Minimum Energy Pathway method when applying the gradient \"force\" to the path",
				"1 Number: the max step, default value = " .. gradientMaxMotion(),
				""
	end

	if x == gradientMaxMotion then
		return "Get the max avrg motion relative to moment strength used in the Minimum Energy Pathway method when applying the gradient \"force\" to the path",
				"",
				"1 Number: the max step, default value = " .. gradientMaxMotion()
	end
	
	if x == pathEnergy then
		return "Get all energies along the path",
				"",
				"1 Table: energies along the path"
	end
	
	-- calling fallback
	if x == nil then
		return help()
	end
	return help(x)
end
