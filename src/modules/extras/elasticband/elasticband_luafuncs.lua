-- ElasticBand

local MODNAME = "ElasticBand"
local MODTAB = _G[MODNAME]
local mt = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local help = MODTAB.help -- setting up fallback to C++ help defs

local function get_eb_data(eb)
	if eb == nil then
		return {}
	end
	if eb:getInternalData() == nil then
		eb:setInternalData({})
	end
	return eb:getInternalData()
end

-- write a state to a spinsystem
local function writePathPointTo(eb, path_point, ssNew)
	local d = get_eb_data(eb)
	local ss = d.ss

	ss:copySpinsTo(ssNew)
-- 	print("pp", path_point)
	local sites = eb:sites()
	for i=1,table.maxn(sites) do
		local x,y,z = eb:spin(path_point, i)
		print(table.concat(sites[i], ","))
		ssNew:setSpin(sites[i], {x,y,z})
	end
end

local function pathEnergy(eb)
	local d = get_eb_data(eb)
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

	eb:calculateEnergies(get_site_ss, set_site_ss, get_energy_ss);
	return eb:getPathEnergy()
end

local function initialize(eb, noise)
	local d = get_eb_data(eb)
	noise = noise or 0.05
	local np = d.np or 20

	eb:resampleStateXYZPath(np/8+2, noise)
	eb:resampleStateXYZPath(np/4+2, noise)
	eb:resampleStateXYZPath(np/2+2, noise)
	eb:resampleStateXYZPath(np, noise)
	
	d.isInitialized = true
end

local function compute(eb, n)
	local d = get_eb_data(eb)
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
		initialize(eb)
	end

	local movement = 0
	n = n or 50
	local dt = eb:gradientMaxMotion()
	for i=1,n do
		eb:calculateEnergyGradients(get_site_ss, set_site_ss, get_energy_ss, 1e-5)
		eb:makeForcePerpendicularToPath()
		eb:makeForcePerpendicularToSpins()
		movement = eb:applyForces(dt)
		eb:resampleStateXYZPath(np)
	end
	
	return movement / np
end

local function setSpinSystem(eb, ss)
	local d = get_eb_data(eb)
	if getmetatable(ss) ~= SpinSystem.metatable() then
		error("setSpinSystem requires a SpinSystem", 2)
	end
	d.ss = ss
end


local function setNumberOfPathPoints(eb, n)
	local d = get_eb_data(eb)
	if type(n) ~= "number" then
		error("setNumberOfPathPoints requires a number", 2)
	end
	if n < 2 then
		error("Number of points must be 2 or greater.")
	end
	d.np = n
end

local function setGradientMaxMotion(eb, dt)
	local d = get_eb_data(eb)
	if type(dt) ~= "number" then
		error("setGradientMaxMotion requires a number", 2)
	end
	d.gdt = dt
end


local function gradientMaxMotion(eb)
	local d = get_eb_data(eb)
	return d.gdt or 0.1
end


local function setEnergyFunction(eb, func)
	local d = get_eb_data(eb)
	if type(func) ~= "function" then
		error("setEnergyFunction requires a function", 2)
	end
	d.energy_function = func
end

local function setSites(eb, tt)
	if type(tt) ~= "table" then
		error("setSites requires a table of 1,2 or 3 Component Tables representing sites.") 
	end
	
	if tt[1] and type(tt[1]) ~= "table" then
		error("setSites requires a table of 1,2 or 3 Component Tables representing sites.") 
	end
	
	eb:clearSites()
	for k,v in pairs(tt) do
		eb:_addSite(v[1], v[2], v[3])
	end
end

local function setInitialPath(eb, pp)
	local msg = "setInitialPath requires a Table of Tables of site orientations"
	local tableType = type({})
	if type(pp) ~= tableType then
		error(msg)
	end
	local numSites = nil
	eb:clearPath()
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
			eb:_addStateXYZ(x,y,z)
		end
	end
end

local function getSpinSystem(eb)
	local d = get_eb_data(eb)
	return d.ss
end

local function getEnergyFunction(eb)
	local d = get_eb_data(eb)
	return (d.energy_function or function() return 0 end)
end

local function getNumberOfPathPoints(eb)
	local d = get_eb_data(eb)
	return (d.np or 20)
end

local function getNumberSites(eb)
	return table.maxn(eb:sites())
end


local function relaxSinglePoint(eb, pointNum, stepSize, epsilon, numSteps)
	local d = get_eb_data(eb)
	local ss = getSpinSystem(eb)
	local energy_function = getEnergyFunction(eb)

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
	-- these or X are for default values
	return eb:_relaxSinglePoint(pointNum or 0, get_site_ss, set_site_ss, get_energy_ss, stepSize or 0.1, numSteps or 1, epsilon or 1e-3)
end

local function computePointSecondDerivative(eb, pointNum, h, destArray)
	local d = get_eb_data(eb)
	local ss = getSpinSystem(eb)
	local energy_function = getEnergyFunction(eb)
	
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
	
	
	local c = eb:siteCount()
	
	if destArray then
		if destArray:nx() ~= c*3 or destArray:ny() ~= c*3 then
			error("Destination array size mismatch. Expected " .. c*3 .. "x" .. c*3)
		end
	else
		destArray = Array.Double.new(c*3, c*3)
	end

	local t = eb:_computePointSecondDerivative(pointNum, h, get_site_ss, set_site_ss, get_energy_ss)

	for x=0,c*3-1 do
		for y=0,c*3-1 do
			destArray:set(x+1, y+1, t[x+y*(c*3)+1])
		end
	end
			
	return destArray
end

local function maximalPoints(eb)
	pathEnergy(eb) -- calculate energies
	return eb:_maximalPoints()
end

local function energyBarrier(eb)
	local mins, maxs, all = eb:maximalPoints()
	
	local ee = eb:pathEnergy()
	
	local start_e = ee[1]
	
	local max_e = start_e
	
	for k,p in pairs(maxs) do
		if ee[p] > max_e then
			max_e = ee[p]
		end
	end
	
	return max_e - start_e
end



mt.setEnergyFunction = setEnergyFunction
mt.setSites = setSites
mt.setNumberOfPathPoints = setNumberOfPathPoints

mt.energyFunction = getEnergyFunction
mt.numberOfPathPoints = getNumberOfPathPoints
mt.numberOfSites = getNumberSites

mt.compute = compute
mt.pathEnergy = pathEnergy

mt.writePathPointTo = writePathPointTo
mt.initialize = initialize

mt.setInitialPath = setInitialPath

mt.setSpinSystem = setSpinSystem

mt.relaxSinglePoint = relaxSinglePoint

mt.hessianAtPoint = computePointSecondDerivative


mt.setGradientMaxMotion = setGradientMaxMotion
mt.gradientMaxMotion = gradientMaxMotion

mt.energyBarrier = energyBarrier

mt.maximalPoints = maximalPoints


MODTAB.help =
function(x)
	if x == setSpinSystem then
		return "Set the SpinSystem that will be used to do energy and orientation calculations. The any changes made to this SpinSystem will be undone before control is returned to the calling environment.",
				"1 *SpinSystem*: SpinSystem to be used in calculations",
				""
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
			"1 Integer, 1 *SpinSystem*: The integer ranges from 1 to the number of path points, the *SpinSystem* will have the sites involved in the Elastic Band calculation changed to match those at the given path index.",
			""
	end
	if x == initialize then
		return 
			"Expand the endpoints into a coherent rotation over the number of path points specified with setNumberOfPathPoints.",
			"1 Optional Number: The magnitude of the noise introduced in the first subdivision of the path into 3 points, default 0.05",
			""
	end
	if x == compute then
		return
			"Run several relaxation steps of the elastic band method",
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
	
	
	if x == setInitialPath then
		return
			"Set the initial path for the elastic band calculation",
			"1 Table of Tables of site orientations: Must be at least 2 elements long to define the start and end points. Example:\n<pre>upup     = {{0,0,1},{0,0,1}}\ndowndown = {{0,0,-1},{0,0,-1}}\n eb:setInitialPath({upup,downdown})\n</pre>",
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
				"1 Integer, 2 Numbers, 1 Integer: Point to relax, step size (positive for down, negative for up), epsilon used to calculate energy gradients, number of iterations.",
				"3 Numbers: Initial Energy, New energy and absolute energy change"
	end
	
	if x == computePointSecondDerivative then
		return "Compute the 2nd order partial derivative at a given point along the path.",
				"1 Integer, 1 Number, 1 Optional Array: Point to calculate 2nd derivative about, step size used in numerical differentiation, optional destination array. If no array is provided one will be created",
				"1 Array: Derivatives"
	end
		
	
	if x == setGradientMaxMotion then
		return "Set the max avrg motion relative to moment strength used in the elastic band method when applying the gradient \"force\" to the path",
				"1 Number: the max step, default value = " .. gradientMaxMotion(),
				""
	end

	if x == gradientMaxMotion then
		return "Get the max avrg motion relative to moment strength used in the elastic band method when applying the gradient \"force\" to the path",
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
