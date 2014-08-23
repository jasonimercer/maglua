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
local function build_gse_closures(mep, JSON)
    local d = get_mep_data(mep)
    local ss = mep:spinSystem()
    local energy_function = mep:energyFunction()
    JSON = JSON or {}
    local report = JSON.report or function()end
    --print("BUILD GSE CLOSURES")
    local function get_site_ss(x,y,z)
	local sx,sy,sz = ss:spin(x,y,z)
	--report(string.format("get(%d,%d,%d)", x,y,z))
	return sx,sy,sz
    end
    local function set_site_ss(x,y,z,sx,sy,sz) --return if a change was made
	local ox,oy,oz,m = ss:spin({x,y,z})
	ss:setSpin({x,y,z}, {sx,sy,sz}, m)
	--report(string.format("set(%d,%d,%d,  %e,%e,%e)", x,y,z,sx,sy,sz))
	return (ox ~= sx) or (oy ~= sy) or (oz ~= sz)
    end
    local function get_energy_ss()
	--  print("energy")
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

local function sweep_level(mep, rules, func, opt_func, depth)
    depth = depth or 1

    if table.maxn(rules) < depth then
	error("Empty rules?")
    end

    local point = rules[depth][1]
    local site  = rules[depth][2]
    local coord = rules[depth][3]

    local start = rules[depth][4]
    local step  = rules[depth][6]
    local endv  = rules[depth][5]

    -- record original
    local ox, oy, oz, oc = mep:_nativeSpin(point, site)

    for v=start,endv,step do
	local x1,x2,x3,c = mep:_nativeSpin(point, site)
	local x = {x1,x2,x3}
	x[coord] = v
	x1,x2,x3 = x[1], x[2], x[3]
	-- print("SET", point, site, x1,x2,x3,c)
	mep:setSpinInCoordinateSystem(point, site, x1,x2,x3,c)
	if depth == table.maxn(rules) then
	    func(mep)
	end

	if depth < table.maxn(rules) then
	    sweep_level(mep, rules, func, opt_func, depth+1)
	end
    end

    opt_func(depth)

    -- restore original
    mep:setSpinInCoordinateSystem(point, site, ox,oy,oz,oc)
end

methods["sweepValues"] = 
    {
    "Sweep coordinates at points and sites calling a function each time.",
    "1 Table of sweep rules, 1 function, 1 optional function: Each rule is an added dimension to the sweep. Each rule specifies a point (1 to mep:numberofPoints()), a site (1 to mep:numberOfSites()), a coordinate (1 to 3), a start value, an end value and a step value. The function will be called on the MEP for each sweep. If the 2nd optional function is specified it will be called after each sweep with the depth of the current sweep.",
    "",
    function( mep, rules, func, opt_func)
	opt_func = opt_func or (function() end)
	sweep_level(mep, rules, func, opt_func, 1)
    end
}

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
	
	local cfg = {}
	local ns = mep:numberOfSites() 
	local sites = mep:sites()

	for i=1,ns do
	    local x,y,z = mep:spin(path_point, i)
	    cfg[i] = {x,y,z}
	end

	ss:copySpinsTo(ssNew)
	local sites = mep:sites()
	for i=1,ns do
	    --print(table.concat(sites[i], ","), table.concat(cfg[i], ","))
	    ssNew:setSpin(sites[i], cfg[i])
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
	local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep, {report=print})
	mep:calculateEnergies(get_site_ss, set_site_ss, get_energy_ss)
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

methods["copy"] = 
    {
    "Create a copy of the MEP object",
    "",
    "1 MEP oject: Copy of calling object",
    function(mep)

	local d = get_mep_data(mep)
	local mep2 = MEP.new()
	mep:internalCopyTo(mep2)

	local id = {}
	for k,v in pairs(d) do
	    id[k] = v
	end
	mep2:setInternalData(id)

	return mep2		
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

methods["resamplePath"] =
    {
    "Resample path using existing number of points or given number of points",
    "0 or 1 Integers: Current or new number of path points",
    "",
    function(mep, _np)
	local np = _np or mep:numberOfPathPoints()
	mep:resampleStateXYZPath(np)
	-- mep:setNumberOfPathPoints(np)
    end
}


local setNumberOfPathPoints_warning = true
methods["setNumberOfPathPoints"] =
    {
    "Backward compatibility method. Gives a warning and calls :resamplePath()",
    "1 Integer: New number of path points",
    "",
    function(mep, _np)
	if setNumberOfPathPoints_warning then
	    setNumberOfPathPoints_warning = false
	    local level = 2
	    local trace = {}
	    local env = debug.getinfo(level)

	    while env do
		table.insert(trace, (env.short_src or "") .. ":" .. (env.currentline or 0))
		level = level + 1
		env = debug.getinfo(level)
	    end

	    -- not showing trace of bootstrap
	    for i=1,4 do
		table.remove(trace)
	    end

	    trace[1] = trace[1] or "Unknown"

	    local warning = "WARNING: This script is using the obsolete method `:setNumberOfPathPoints(x)' at: (" .. trace[1] .. ")"
	    warning = warning .. "\nWARNING: Please replace `:setNumberOfPathPoints(x)' with `:resamplePath(x)'."

	    io.stderr:write(warning .. "\n")
	end

	return mep:resamplePath(_np)
    end
}


methods["equalPoints"] =
{
"Test if two points are equal. Equality is determined by having angles between all points less than a given tolerance (in radians)",
"2 Integers, 1 Optional Number: Point indices, optional tolerance or 5 degrees expressed in radians",
"1 Boolean: Result of equality",
function(mep, idx1, idx2, tol)
    tol = tol or 5 * math.pi / 180

    for k,v in pairs(mep:anglesBetweenPoints(idx1, idx2)) do
	if v > tol then
	    return false
	end
    end
    return true
end
}

methods["equal"] =
    {
    "Test if points or sites are equal. Equality if determined by comparing the angle between vectors.",
    "2 Integers or 2 Tables each of 2 Integers, 1 optional Number: If 2 Tables of 2 Integers are given then they will each be interpreted as a point and site index and the two sites at the two points will be compared. If 2 Integers are given then they will be interpreted as points and all the sites at those points will be compared. If a number is provided it will be used a the maximum number of radians that two vectors can differ by and still be considered equal, default is 5 degrees converted to radians.",
    "1 Boolean: Test result",
    function(mep, v1,v2,tol)
	tol = tol or 5 * math.pi / 180

	if type(v1) ~= type(v2) then
	    error("first two arguments must be the same type")
	end

	if v1 == nil then
	    error("nil input")
	end

	if type(v1) == type(1) then -- integers
	    for s=1,mep:numberOfSites() do
		if not mep:equal({v1,s}, {v2,s}, tol) then
		    return false
		end
	    end
	    return true
	end

	if type(v1) == type({}) then
	    local p1, s1, p2, s2 = v1[1], v1[2], v2[1], v2[2]
	    local angle = mep:_angleBetweenPointSite(p1,s1,p2,s2)
	    --print(p1,s1,p2,s2,angle,tol,angle<=tol)
	    return angle <= tol
	end

	error("Invalid input")
    end
}

methods["simplifyPath"] =
    {
    "Remove cycles in a path. If a path goes from A to B to C to A to B then the path will be simplified to A to B. If a path goes from A to A to B then the path will be simplified to A to B.",
    "1 Optional number: The maximum angle allowed between two vectors while still considering them equal. If none is supplied then 5 degrees expressed in radians is used.",
    "",
    function(mep, tol)
	tol = tol or (5 * math.pi / 180)

	-- we will march through the path. At each site we will look ahead at all other sites and see if it 
	-- is the same as the current site. If so we will cull that and all sites in between. If we do that 
	-- we will restart the algorithm until no more duplicates have been found
	local np = mep:numberOfPoints()
	for p=1,np-1 do
	    for q=np,p+1,-1 do
		if mep:equalPoints(p,q,tol) then -- we have a cycle
		    --print("points ", p,q, " are equal")
		    local bad_points = {}
		    if q == mep:numberOfPoints() then
			if p > 1 then
			    table.insert(bad_points, p)
			end
		    end
		    for j=p+1,q do
			if j < mep:numberOfPoints() then -- never ever ever delete the last point... ever
			    table.insert(bad_points, j)
			end
		    end

		    if table.maxn(bad_points) > 0 then -- may be zero if trying to remove last point
			--print("Deleting " .. table.concat(bad_points))
			mep:deletePoints(bad_points)
			return mep:simplifyPath(tol) -- restart with a tail call
		    end
		end
	    end
	end
    end
}


methods["initialize"] = 
    {
    "Create child MEPs for adaptive stepping. This is an internal function and is called automaticaly.",
    "",
    "",
    function(mep)
	local d = get_mep_data(mep)

	if not mep:isChild() then
	    if d.big_step == nil then
		d.big_step = MEP.new()
		d.small_step = MEP.new()
		
		d.big_step:setChild(true)
		d.small_step:setChild(true)
	    end
	end
    end
}


methods["movingSites"] =
    {
    "Determine which sites are in motion along the path.",
    "",
    "2 Table of Integers: Table of indices of sites which move along the path, table of ndices of sites which are stationary along the path.",
    function(mep)
	local movers = {}
	local stationary = {}
	local same  = {}
	for s=1,mep:numberOfSites() do
	    same[s] = true
	end

	for p=2,mep:numberOfPoints() do
	    local a = mep:anglesBetweenPoints(1, p)
	    for s=1,mep:numberOfSites() do
		if a[s] > 1e-10 then
		    same[s] = false
		end
	    end
	end

	for k,v in pairs(same) do
	    if v == true then
		table.insert(movers, k)
	    else
		table.insert(stationary, k)
	    end
	end
	return movers, stationary
    end

}

--[[
methods["describePoint"] = 
    {
    "Get a human-readable description of a point",
    "1 Integer, 1 Optional String: point index and custom Coordinate System, default is the current coordinate system at the point",
    "1 Table of Strings: {Site Description, Orientation Description}",
    function(mep, p, cs)
	local txt = {}
	for s=1,mep:numberOfSites() do
	    local x,y,z,c = mep:spinInCoordinateSystem(p,s,cs)
	    txt[s] = "(" .. table.concat({x,y,z,"`"..c.."'"}, ", ") .. ")"
	end
	txt = table.concat(txt, ", ")

	local sites = mep:sites()
	for k,v in pairs(sites) do
	    sites[k] = "{" .. table.concat(v, ", ") .. "}"
	end
	local txtSites = "{" .. table.concat(sites, ", ") .. "}"

	return {"Site: " .. txtSites, "Orientation: " .. txt}
	
    end
}
--]]

methods["reverse"] =
    {
    "Reverse the path",
    "",
    "",
    function(mep)
	local path1 = mep:path()
	local path2 = {}
	local np = mep:numberOfPoints()
	for i=1,np do
	    path2[np-i+1] = path1[i]
	end
	mep:setInitialPath(path2)
    end
}


local function single_compute_step(mep, get_site_ss, set_site_ss, get_energy_ss)
    local d = get_mep_data(mep)

    mep:initialize()

    mep:calculateEnergyGradients(get_site_ss, set_site_ss, get_energy_ss)
    --mep:makeForcePerpendicularToPath(get_site_ss, set_site_ss, get_energy_ss)
    --mep:makeForcePerpendicularToSpins(get_site_ss, set_site_ss, get_energy_ss)
    mep:applyForces()
end

local function filter_arg(arg)
    local result = {}
    result[ type(1) ] = {}
    result[ type("s") ] = {}
    result[ type({}) ] = {}
    result[ type(print) ] = {}
    for i=1,table.maxn(arg) do
	result[ type(arg[i]) ] = result[ type(arg[i]) ] or {}
	table.insert(result[ type(arg[i]) ], arg[i])
    end
    return result
end

methods["compute"] =
    {
    "Run several relaxation steps of the Minimum Energy Pathway method",
    "1 Optional Integer, 1 Optional Number: Number of steps, default 50. Tolerance different than tolerance specified. Optional JSON-style table: report key with a print function.",
    "1 Number: Number of successful steps taken, for tol > 0 this may be less than number of steps requested",
    function(mep, ...)
	local results = filter_arg(arg)

	local n    = results[ type(1) ][1] or 50 
	local tol  = results[ type(1) ][2] or mep:tolerance()
	
	local first_tab = results[ type({}) ][1] or {}

	local report = first_tab["report"] or function() end
	
	local d = get_mep_data(mep)
	local ss = mep:spinSystem()
	local np = mep:numberOfPathPoints()
	local energy_function = d.energy_function
	
	if ss == nil then
	    error("SpinSystem is nil. Set a working SpinSystem with :setSpinSystem")
	end
	
	if energy_function == nil then
	    error("Energy function is nil. Set an energy function with :setEnergyFunction")
	end
	
	local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
	
	mep:initialize()
	
	local successful_steps = 0
	for i = 1,n do
	    local current_beta = mep:beta()
	    
	    -- first bool below is for "do_big" second is for "do_small"
	    copy_to_children(mep, true, tol>=0)
	    
	    d.big_step:setBeta(current_beta)
	    single_compute_step(d.big_step, get_site_ss, set_site_ss, get_energy_ss)
	    
	    if tol > 0 then -- negative tolerance will indicate no adaptive steps
		d.small_step:setBeta(current_beta/2)
		
		--print("small step 1:")
		single_compute_step(d.small_step, get_site_ss, set_site_ss, get_energy_ss)
		--print("small step 2:")
		single_compute_step(d.small_step, get_site_ss, set_site_ss, get_energy_ss)
		
		local aDiff, maxDiff, max_idx = d.big_step:absoluteDifference(d.small_step)
		local aDiffAvrg = aDiff / np
		
		-- print("beta = ", current_beta)
		local step_mod, good_step = getStepMod(tol, maxDiff)
		--print("step_mod = ", step_mod)
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
	    report(string.format("step %3d of %3d, step size: %e, energy barrier: %e", i, n, mep:beta(), mep:energyBarrier()))
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

methods["resize"] =
    {
    "Set the number of path points and/or the number of sites without interpolating. This method should be used carefully.",
    "1 or 2 Integers: New number of path points and sites. Missing values will be the existing values.",
    "",
    function(mep, npp, ns)
	npp = npp or mep:numberOfPathPoints()
	ns = ns or mep:numberOfSites()

	mep:_resize(npp, ns)

	-- mep:setNumberOfPathPoints(npp)
    end
}

methods["point"] =
    {
    "Get the coordinates for a given point.",
    "1 Integer: Point index.",
    "1 Table of sites: Each site is a table of 3 numbers and a string where the numbers represent a vector and the string is a coordinate system name.",
    function(mep, p)
	if p < 1 or p > mep:numberOfPoints() then
	    error("Invalid point")
	end
	local t = {}
	for s=1,mep:numberOfSites() do
	    local x1,x2,x3,cs = mep:spinInCoordinateSystem(p,s)
	    t[s] = {x1,x2,x3,cs}
	end
	return t
    end
}

methods["pointInCoordinateSystem"] =
    {
    "Get the coordinates for a given point in a custom Coordinate system.",
    "1 Integer, 1 String: Point index, coordinate system name",
    "1 Table of sites: Each site is a table of 3 numbers and a string where the numbers represent a vector and the string is a coordinate system name.",
    function(mep, p, _cs)
	if p < 1 or p > mep:numberOfPoints() then
	    error("Invalid point")
	end
	local t = {}
	for s=1,mep:numberOfSites() do
	    local x1,x2,x3,cs = mep:spinInCoordinateSystem(p,s,_cs)
	    t[s] = {x1,x2,x3,cs}
	end
	return t
    end
}

methods["setPoint"] = 
    {
    "Set the coordinates for a given point.",
    "1 Integer, 1 Table of sites: Point index, Each site is a table of 3 numbers and a string where the numbers represent a vector and an optional string representing coordinate system name (default Cartesian).",
    "",
    function(mep, p, pt)
	if p < 1 or p > mep:numberOfPoints() then
	    error("Invalid point")
	end

	for s=1,mep:numberOfSites() do
	    local x1,x2,x3,c = pt[s][1], pt[s][2], pt[s][3], pt[s][4] or "Cartesian"
	    mep:setSpinInCoordinateSystem(x1,x2,x3,c)
	end
    end
}

methods["path"] =
    {
    "Get the current path.",
    "",
    "Table of points. Each point is a table of sites. Each site is a table of 3 numbers and a string where the numbers represent a vector and the string is a coordinate system name. This return value is compatible with setInitialPath.",
    function(mep)
	local t = {}
	for p=1,mep:numberOfPoints() do
	    t[p] = mep:point(p)
	end
	return t
    end
}

methods["keepPoints"] =
    {
    "Compliment of deletePoints. Delete points not provided. Synonym for reduceToPoints",
    "N Integers or 1 Table of integers: The points to leep while deleting the others.",
    "",
    function(mep, ...)
	if type(arg[1]) == type({}) then -- we were given a table
	    arg = arg[1]
	end

	local keepers = arg
	table.sort(keepers)
	
	local old_path = mep:path()

	local new_path = {}
	for i=1,table.maxn(keepers) do
	    table.insert(new_path, old_path[keepers[i]])
	end
	
	mep:setInitialPath(new_path)
    end
}

methods["deletePoints"] =
    {
    "Delete points at the given indices.",
    "N Integers or 1 Table of integers: The points to delete while keeping the others.",
    "",
    function(mep, ...)
	if type(arg[1]) == type({}) then -- we were given a table
	    arg = arg[1]
	end

	local keepers = {}
	for p=1,mep:numberOfPoints() do
	    keepers[p] = p
	end

	table.sort(arg)

	for i=table.maxn(arg),1,-1 do
	    table.remove(keepers, arg[i])
	end

	mep:keepPoints(keepers)
    end
}


methods["copyPointTo"] =
    {
    "Copy configuration at source indices to destination indices.",
    "2 Integers or 2 Tables of Integers",
    "",
    function(mep, a,b)
	if type(a) == type(1) then
	    a = {a}
	end
	if type(b) == type(1) then
	    b = {b}
	end
	return mep:_copy(a,b)
    end
}

methods["swapPoints"] =
    {
    "Swap configuration at source indices and destination indices.",
    "2 Integers or 2 Tables of Integers",
    "",
    function(mep, a,b)
	if type(a) == type(1) then
	    a = {a}
	end
	if type(b) == type(1) then
	    b = {b}
	end
	return mep:_swap(a,b)
    end
}



--[[
methods["pointNearSingularity"] =
    {
    "Check if a vector at a point and site is near a singularity either in the given coordinate system or in the current coordinate system.",
    "2 Integers, 1 optional ",
    "",
    function(mep, point, site, cs)
	cs = cs or mep:coordinateSystem()

	local x,y,z = mep:spin(point, site)
	local r = (x^2 + y^2 + z^2)^(1/2)
	if r == 0 then
	    return true
	end
	-- "Cartesian", "Spherical", "Canonical", "SphericalX", "SphericalY", "CanonicalX", "CanonicalY"
	local v = {x/r, y/r, z/r}

	if type(a) == type(1) then
	    a = {a}
	end
	if type(b) == type(1) then
	    b = {b}
	end
	return mep:_copy(a,b)
    end
}
--]]


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
	
	-- need to clear children so they get reinitialized
        local d = get_mep_data(mep)
	d.big_step = nil

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
    "1 Table of Tables of site orientations or nils: Must be at least 2 elements long to define the start and end points. Example:\n<pre>upupup     = {{0,0,1}, {0,0,1}, {0,0,1}}\ndowndowndc = {{0,0,-1},{0,0,-1},nil}\n mep:setInitialPath({upupup,downdowndc})\n</pre>Values of nil for orientations in the start or end points mean that the algorithm will not attempt to keep them stationary - they will be allowed to drift. Their initial value will be whatever they are in the SpinSystem at the time the method is called. These drifting endpoints are sometimes referred to as `don't care' sites. Coordinate systems are Cartesian by default but a 4th table value of a string can specify the coordinate system.",
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
		local c = "Cartesian"
		if pp[p][s] == nil then --user doesn't care
		    mep:_setImageSiteMobility(p, s, 1)
		else -- user gave explicit orientations
		    x = pp[p][s][1] or x
		    y = pp[p][s][2] or y
		    z = pp[p][s][3] or z
		    c = pp[p][s][4] or "Cartesian"
		    mep:_setImageSiteMobility(p, s, mobility)
		end
		mep:_addStateXYZ(x,y,z,c)
	    end
	end
    end
}



methods["numberOfPathPoints"] =
    {
    "Get the number of path points used to approximate a line.",
    "",
    "1 Integer: Number of path points",
    function(mep)
	return mep:pathPointCount()
    end
}

methods["numberOfPoints"] =
    {
    "Synonym for :numberOfPathPoints()",
    methods["numberOfPathPoints"][2],
    methods["numberOfPathPoints"][3],
    methods["numberOfPathPoints"][4],
}



methods["numberOfSites"] = 
    {
    "Get the number of sites used in calculation.",
    "",
    "1 Integer: Number of sites.",
    function(mep)
	return mep:siteCount()
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
	
        local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
	
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
    
    local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)

    steps = steps or 10
    goal = goal or 0
    local h, good_steps = nil
    for i=1,steps do
	min_mag_grad, h, good_steps = mep:_relaxSinglePoint_sd(pointNum, get_site_ss, set_site_ss, get_energy_ss, h, goal)
	-- print(min_mag_grad, "<", goal)
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

methods["expensiveEnergyMinimizationAtPoint"] = 
    {
    "Attempt to minimize energy at a point using inefficient methods",
    "1 Integer, 1 Optional Integer, 1 Optional Number: Point index, number of steps (default 50), starting step size (default 1e-3)",
    "1 Integer, 3 Number: Ratio of successful steps to total steps, initial energy, final energy, final step size",
    function(mep, point, steps, h)
	local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
	
	steps = steps or 50

	local good_steps, iE, fE, fH = mep:_expensiveEnergyMinimization(get_site_ss, set_site_ss, get_energy_ss, point, (h or 1e-3), steps)

	return good_steps / steps, iE, fE, fH
    end
}

methods["minimizeEnergyOfCustomConfiguration"] =
    {
    "Attempt to minimize energy of a custom inputted configuration. This method will temporarily replace a point in the current path with a custom configuration, minimize it, record it, restore the old configurations and return the minimized configuration",
    "1 Table of Tables, 1 Optional Integer, 1 Optional number: Custom initial orientations for the sites, each vector is a table of 3 numbers and 1 optional Coordinate System name. First optional integer is the number of steps (default 50) second optional value is the starting step size (default 1e-3).",
    "1 Table of Tables, 1 Number: result of energy minimization on input configuration, energy at final state.",
    function(_mep, cfg, steps, h)
	local mep = _mep:copy()

	mep:setInitialPath({cfg,cfg})
	local cs = mep:coordinateSystem(1,1)
	mep:setCoordinateSystem("Spherical")
	mep:expensiveEnergyMinimizationAtPoint(1, steps, h)
	mep:setCoordinateSystem(cs)
	local res = {}
	for s=1,mep:numberOfSites() do
	    local x,y,z,c = mep:spinInCoordinateSystem(1,s)
	    res[s] = {x,y,z,c}
	end
	return res, mep:energyAtPoint(1)
    end

}

methods["energyOfCustomConfiguration"] =
    {
    "Compute the energy of the custom given configuration",
    "1 Table of Tables: Custom orientations for the sites, each vector is a table of 3 numbers and 1 optional Coordinate System name (Default Cartesian).",
    "1 Number: energy of custom configuration.",
    function(_mep, cfg)
	local mep = _mep:copy()
	mep:setInitialPath({cfg,cfg})
	return mep:energyAtPoint(1)
    end
}


methods["interpolateBetweenCustomPoints"] =
    {
    "Interpolate between two custom points.",
    "2 Tables of Vectors, 1 Number: Each Vector is defined as 3 numbers and 1 optional string stating the coordinate system (Default Cartesian). The number is a ratio ideally between 0 and 1 which interpolates between the two input.",
    "1 Table of Vectors: The interpolated point.",
    function(mep, p1, p2, r)
	return mep:_interpolateBetweenCustomPoints(p1, p2, r)
    end
}



local phi = (1 + math.sqrt(5)) / 2
local resphi = 2 - phi

local goldenSectionSearch = nil
local function goldenSectionSearch(a,b,c, fa,fb,fc, tau, f)
    local x
    if (c - b) > (b - a) then
	x = b + resphi * (c - b)
    else
	x = b - resphi * (b - a)
    end

    if (math.abs(c - a) < tau * (math.abs(b) + math.abs(x))) then
	return (c + a) / 2
    end

    local fx = f(x)
    
    --assert(f(x) != f(b));
    if (fx < fb) then 
	if (c - b > b - a) then
	    return goldenSectionSearch(b,x,c, fb,fx,fc, tau, f)
	else 
	    return goldenSectionSearch(a,x,b, fa,fx,fb, tau, f)
	end
    else 
	if (c - b > b - a) then
	    return goldenSectionSearch(a,b,x, fa,fb,fx, tau, f)
	else
	    return goldenSectionSearch(x,b,c, fx,fb,fc, tau, f)
	end
    end
end


local function boundingMin(mep, i, scale)
    scale = scale or 1
    local e = mep:energyAtPoint(i) * scale

    if i > 1 then
	if i < mep:numberOfPoints() then
	    if mep:energyAtPoint(i-1)*scale < e then
		return i-1,i
	    else
		return i,i+1
	    end
	else
	    return i-1,i
	end
    else
	return i,i+1
    end
end
local function boundingMax(mep,i)
    return boundingMin(mep,i,-1)
end

methods["interpolatedCriticalPoints"] =
{
"Use :interpolateBetweenCustomPoints() and a golden ratio search to find critical points along the path",
"1 Optional numbers: Tolerance, tau as defined in the golder search wikipedia page. Default 1e-8",
"3 Tables of Numbers: Tables of non-integer point along path corresponding to each interpolated set of minimum point, . These Non-integer points can be transformed into points with the :interpolatePoint() method.",
function(mep, tol, steps)
    tol = tol or 1e-8
    steps = steps or 10
    local mins, maxs, all = mep:criticalPoints()
    local imins, imaxs, iall = {}, {}, {}

    local function min_func(x)
	return mep:energyOfCustomConfiguration( mep:interpolatePoint(x ))
    end
    local function max_func(x)
	return mep:energyOfCustomConfiguration( mep:interpolatePoint(x )) * -1
    end
    local tau = tol

    for i,v in pairs(mins) do
	local x1,x3 = boundingMax(mep, v)
	local f = min_func
	local x2 = (x1+x3)/2
	local x2 = goldenSectionSearch(x1,x2,x3, f(x1),f(x2),f(x3), tau, f)
	imins[i] = x2
	table.insert(iall, x2)
    end

    for i,v in pairs(maxs) do
	local x1,x3 = boundingMax(mep, v)
	local f = max_func
	local x2 = (x1+x3)/2
	local x2 = goldenSectionSearch(x1,x2,x3, f(x1),f(x2),f(x3), tau, f)
	imaxs[i] = x2
	table.insert(iall, x2)
    end

    table.sort(iall)
    return imins, imaxs, iall
end
}

methods["interpolatePoint"] =
{
"Interpolate to get point at non-integer point index",
"1 Number: point location along path from 1 to mep:numberOfPoints()",
"1 Table of Vectors: interpolated configuration",
function(mep, p)
    local f = math.floor(p)

    if f == p then
	return mep:point(p)
    end

    if f+1 > mep:numberOfPoints() then
	error("Out of range [1:".. mep:numberOfPoints() .."]")
    end

    local r = p-f

    local cfg1 = mep:point(f)
    local cfg2 = mep:point(f+1)

    return mep:interpolateBetweenCustomPoints(cfg1, cfg2, r)
end
}


methods["expensiveGradientMinimizationAtPoint"] = 
    {
    "Attempt to minimize the energy gradient at a point using inefficient methods",
    "1 Integer, 1 Optional Integer, 1 Optional Number: Point index, number of steps (default 50), starting step size (default 1e-3)",
    "1 Integer, 3 Number: Ratio of successful steps to total steps, initial square of the gradient, final square of the gradient, final step size",
    function(mep, point, steps, h)
	local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
	
        steps = steps or 50

        local good_steps, iG, fG, fH = mep:_expensiveGradientMinimization(get_site_ss, set_site_ss, get_energy_ss, point, (h or 1e-3), steps)

        return good_steps / steps, iG, fG, fH
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
	
	local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
	
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
    function(mep, pointNum, kind, h)
	local d = get_mep_data(mep)
	if d.isInitialized == nil then
	    mep:initialize()
	end
	
	h = h or 1e-4
	local step = {}
	for i=1,mep:numberOfPoints() * 3 do
	    step[i] = h
	end

        local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
	
	local t1, t2, t3 = mep:_classifyPoint(pointNum, step, set_site_ss, get_site_ss, get_energy_ss)
	return t1, t2, t3
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
        local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
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
        local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)

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
        local get_site_ss, set_site_ss, get_energy_ss = build_gse_closures(mep)
	
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
	if s == nil then
	    error("Coordinate system required")
	end
	local data = {}
	data["cartesian"] = {{"x", "y", "z"}, {"X Axis", "Y Axis", "Z Axis"}}
	data["spherical"] = {{"r", "p", "t"}, {"Radial", "Azimuthal", "Polar"}}
	data["canonical"] = {{"r", "phi", "p"},{"Radial", "Azimuthal", "cos(Polar)"}}
	if s == nil then
	    error("method requires a coordinate system name")
	end
	if data[string.lower(s)] == nil then
	    error("method requires a valid coordinate system name, given = `" .. s .. "'")
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

methods["minCount"] = 
    {
    "Convenience function. Counts the number of elements in the minimum points returned from MEP:criticalPoints()",
    "",
    "1 Number: Number of minimum points",
    function(mep)
	local mins, maxs, all = mep:criticalPoints()
	return table.maxn(mins)
    end
}
  

methods["maxCount"] =
    {
    "Convenience function. Counts the number of elements in the maximum points returned from MEP:criticalPoints()",
    "",
    "1 Number: Number of minimum points",
    function(mep)
        local mins, maxs, all = mep:criticalPoints()
        return table.maxn(maxs)
    end
}




methods["reduceToPoints"] = methods["keepPoints"]
--[[
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
		local x1, x2, x3, cs = mep:_nativeSpin(v,j)
		t[j] = {x1,x2,x3,cs}
	    end
	    new_initial_path[i] = t
	    n = n + 1
	end

	mep:setInitialPath(new_initial_path)
	-- mep:setNumberOfPathPoints(n) -- this is done in the above call
    end
}
--]]

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
	    local x, y, z, cs = mep:_nativeSpin(p, s)
	    table.insert(state, {x,y,z,cs})
	end
	table.insert(sub_path, state)
    end
    
    -- smep:setNumberOfPathPoints(p2-p1+1) -- this is done in setInitialPath
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
		local x,y,z,cs = mep:_nativeSpin(p, i) 
		t[i] = {x,y,z,cs}
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

	-- mep:setNumberOfPathPoints(table.maxn(initial_path)) -- done in setInitialPath
	mep:setInitialPath(initial_path)
	mep:initialize()

    end
}

methods["convertCoordinateSystem"] =
    {
    "Convert a vector from a source coordinate system to a destination coordinate system",
    "1 Table of 3 Numbers or 3 Numbers, 2 Strings: A vector given either as a table of nubmers or as 3 numbers and Source and Destination Coordinate System names (must match one" ..
	"of the return values from MEP:coordinateSystems() ).",
    "1 Table of 3 Numbers or 3 Numbers and 1 String: The return transformed vector will be returned in the same format as the input vector and the name of the new coordinate system will be returned as the last argument",

    function(mep, ...)
	return mep:_convertCoordinateSystem(...)
    end
}

local function pointsOnSphere(mep, n, rad, cs)
    cs = cs or "Cartesian"
    rad = rad or 1
    local dlong = math.pi * (3 - math.sqrt(5))
    local dz =  2/n
    local long = 0
    local z =  1 - dz/2
    local t = {}
    for k=1,n do
	if z < -1 then
	    z = 1
	end
	local r = math.sqrt(1-z^2)
	local x1,x2,x3,cc = mep:convertCoordinateSystem(math.cos(long)*r*rad, math.sin(long)*r*rad, z*rad, "Cartesian", cs)
	local p = {x1,x2,x3,cc}
	table.insert(t, p)
        z    = z - dz
        long = long + dlong
    end
    return t
end

-- increment multidimensional value in t with max-value n
local function incd(t, n, i)
    i = i or 1
    t[i] = (t[i] or 0) + 1
    if t[i] == n+1 then
	t[i] = 1
	return incd(t,n,i+1)
    end
    return t
end

local function getFromTables(src, idxs)
    local t = {}
    for i=1,table.maxn(idxs) do
	table.insert(t, src[i][idxs[i]])  
    end
    return t
end

methods["evenlyDistributedPoints"] = 
    {
    "Get a list of points evenly distributed over the sites. Each point is a table of orientations.",
    "1 Integer, 1 Optional string: Number of points, name of coordinate system (default Cartesian).",
    "1 Table: Each table element is a table of orientations.",
    function(mep, n, cs)
	-- the plan is to get an even distribution of points on each 
	-- site and then combine them to get even distribution over all
	-- sites

	local list = {}

	local ns = mep:numberOfSites()
	local ss = mep:spinSystem()
	local sites = mep:sites()
	local points_per_sphere = math.ceil(n ^ (1/ns))
	local points = {}
	for i=1,ns do
	    local magnitude = ss:spinArrayM():get(sites[i])
	    local r = pointsOnSphere(mep, points_per_sphere, magnitude, cs)
	    points[i] = r
	end

	-- we now have an even distribution of points for each site stored in the array "points"
	local idx = {}
	for i=1,ns do
	    table.insert(idx, 1)
	end
	for i=1,points_per_sphere^ns do
	    table.insert(list, getFromTables(points, idx))
	    idx = incd(idx, points_per_sphere)
	end

	-- cut list down to requested size (rather than something to the power of the number of sites)
	while table.maxn(list) > n do
	    table.remove(list)
	end

	return list
    end
}



methods["globalRelax"] =
    {
    "Run several relaxation steps where each point takes steps to reduce it's energy. Path resampling is not done. This is similar to :compute() minus the resampling.",
    "1 Optional Integer, 1 Optional Number: Number of steps, default 50. Initial step size for each point.",
    "",
    function(mep, n, h)
	for p=1,mep:numberOfPoints() do
	    mep:expensiveEnergyMinimizationAtPoint(p, n, h)
	end
    end
}

methods["uniquePoints"] =
    {
    "Get a list of which points that are unique.",
    "Zero or 1 numbers, 0 or 1 tables: Tolerances used to determine equality. If a number is provided it will be used. The tolerance is radian difference that two \"equal\" vectors can differ by, default is the default used by the mep:equal() metamethod. If a table is provided then only the points in the table will be considered, otherwise all points will be considered.",
    "1 Table: Indexes of unique points",
    function(mep, ...)
	local nums, tabs = {}, {}
	for k,v in pairs(arg) do
	    if(type(v)) == type(0) then -- number
		table.insert(nums, v)
	    end
	    if(type(v)) == type({}) then -- table
		table.insert(tabs, v)
	    end
	end
	
	local pts = {}
	if tabs[1] == nil then
	    for i=1,mep:numberOfPoints() do
		pts[i] = i
	    end
	    return mep:uniquePoints(nums[1], pts)
	end

	for k,v in pairs(tabs[1]) do -- making a copy of the table
	    pts[k] = v
	end

	local np = table.maxn(pts)

	for i=1,np do
	    for j=i+1,np do
		if pts[j] and pts[i]then
		    if mep:equal(pts[i], pts[j], nums[1]) then
			if pts[i] == 1 then -- we keep 1
			    pts[j] = nil
			else
			    pts[i] = nil
			end
		    end
		end
	    end
	end

	local up = {}
	for i=1,np do
	    if pts[i] then
		table.insert(up, pts[i])
	    end
	end
	return up
    end
}


local function p_cs(mep)
    for p=1,mep:numberOfPoints() do
	local t = {}
	for s=1,mep:numberOfSites() do
	    t[s] = mep:coordinateSystem(p,s)
	end
	print(p, table.concat(t, ", "))
    end
end

methods["printPoint"] = 
    {
    "Primarily a debugging method. Prints a given point.",
    "1 Integer, 1 Optional String: Point index, optional coordinate",
    "",
    function(mep, p, cs) 
	local pe = mep:pathEnergy()

	local t = {string.format("%3d  % 6.6e", p, pe[p])}
	for s=1,mep:numberOfSites() do
	    local x,y,z,c = mep:spinInCoordinateSystem(p,s,cs)
	    table.insert(t, string.format(" {% 4.4e, % 4.4e, % 4.4e, %10s}", x,y,z,c))
	end
	print(table.concat(t))
    end
}

methods["printPath"] = 
    {
    "Primarily a debugging method. Prints the current path.",
    "",
    "",
    function(mep) 
	for p=1,mep:numberOfPoints() do
	    mep:printPoint(p)
	end
    end
}

methods["findMinima"] =
    {
    "Find the coordinates of minima in the energy landscape",
    "1 Integer or input compatible with :setInitialPath(): Number of initial points which will be spread out evenly over hyper-sphere created by the sites involved in the calculation. If a table is instead given then the orientations in it will be used for the search start. Additionally, a JSON-style table can be provided as the second argument with data at a \"report\" key, data at a \"cs\" key and data at a \"hints\" key. The data at the report will be a function that will be called with human readable data regarding the progress of the algorithm (common choice is the print function). The cs key is the coordinate system used in the case that you are not specifying starting points but a point count and the program will evenly distribute them (a common choice, as well as the default, is \"Spherical\"). The hints data contains a list of starting points that should be added to whatever search will take place. ",
    "Table of minima: Each minimum will be a table of sites, each site is 3 numbers representing a vector and a string naming the coordinate system.",
    function(mep, n, json)
	json = json or {}
	if type(n) == type(4) then -- need n points over hypersphere		
	    local cs = json.cs or "Spherical"
	    return mep:findMinima(mep:evenlyDistributedPoints(n, cs), json)
	end
	if type(n) == type({}) then
	    local report = json.report or function() end
	    local hints = json.hints

	    mep:setInitialPath(n)

	    -- adding hint like this so we don't pollute the input
	    if hints then
		local p = mep:path() 
		for k,v in pairs(hints) do
		    table.insert(p,v)
		end
		mep:setInitialPath(p)
	    end
	    
	    mep:setCoordinateSystem( json.cs or "Spherical" )

	    report("Starting search with " .. mep:numberOfPoints() .. " points")

	    local default_plan = {
		-- steps, relax step,  relax steps, equal test
		{      9,      1e-2,           40,     0.0001},
		{      3,      1e-2,           40,     0.0002},
		{      3,      1e-2,           40,     0.005},
 		{      3,      1e-2,           40,     0.3},
 		{      3,      1e-2,          100,     0.2}
 		--{      1,      1e-6,          100,     0.1},
	    }
	    
	    local plan = json.plan or  default_plan
	    local mid_up = nil
	    local num_up
	    for j=1,table.maxn(default_plan) do
		local v = default_plan[j]
		local steps  = v[1]
		local rtol   = v[2]
		local rsteps = v[3]
		local eqtol  = v[4]

		for i=1,steps do
		    -- mep:printPath()
		    mep:globalRelax(rsteps, rtol)
		    -- mep:printPath()
		    local up = mep:uniquePoints(eqtol * math.pi/180)
		    num_up = table.maxn(up)

		    if table.maxn(up) == 0 then
			interactive("Zero unique points")
		    end
		    mep:reduceToPoints(up)
		    -- mep:printPath()
		    report(string.format("Global relax tolerance: %4g, " .. 
					 "Unique test degrees: %4g, Unique points: %4d",
				     rtol, eqtol, table.maxn(up)))
		end
		if j == 3 then
		    mid_up = num_up
		end
	    end
 
	    if mid_up ~= num_up then -- we may still be refining
		json.hints = nil -- clearing hints so this is just a continuation
		report("search has not converged, continuing search with current state")
		return mep:findMinima(mep:path(), json)
	    end

	    -- finalization
	    for i=1,mep:numberOfPoints() do
		local ratio, step_size = 1, 1e-10
		local iteration = 0
		while ratio > 1/30 and iteration < 10 do
		    ratio, _, _, step_size = mep:expensiveEnergyMinimizationAtPoint(i, 30, step_size)
		    iteration = iteration + 1
		end
	    end

	    local up = mep:uniquePoints(2 * math.pi/180)
	    if table.maxn(up) == 0 then
		interactive("Zero unique points")
	    end
	    mep:reduceToPoints(up)

	    return mep:path()
	end
	

	error("`findMinima' expects an integer or table as input")
    end
}


methods["spinInCoordinateSystem"] =
    {
    "Get site as current coordinate system or given coordinate system",
    "2 Integers, Optional String: 1st integer is path index, 2nd integer is site index. Positive values count from the start, negative values count from the end. If a string is given the vector will be returned that that coordinate system",
    "3 Numbers: Vector at site s at path point p.",
    function(mep, ...)
	local ints = {}
	local txts = {}

	for i=1,table.maxn(arg) do
	    if type(arg[i]) == type(1) then
		table.insert(ints, arg[i])
	    end
	    if type(arg[i]) == type("s") then
		table.insert(txts, arg[i])
	    end
	end

	local point = ints[1] or 0

	if point == 0 then
	    error("Invalid point: mep:spinInCoordinateSystem(" .. table.concat(arg, ",") .. ")")
	end

	if point < 0 then
	    point = mep:numberOfPathPoints() + point + 1
	end

	local site = ints[2] or 0
	if site == 0 then
	    error("Invalid site")
	end
	if site < 0 then
	    site = mep:numberOfSites() + site + 1
	end
	local x1, x2, x3, cs = mep:_nativeSpin(point, site)
	
	--print("NS", x1,x2,x3,cs)
	--local cs = mep:coordinateSystem(point,site)
	-- print("CCS", x1,x2,x3,cs, txts[1] or cs)
	
	return mep:convertCoordinateSystem(x1,x2,x3,cs, txts[1] or cs)

    end
}

methods["setPointSiteMobility"] =
    {
    "Usually the endpoints are fixed for the MEP algorithm. This allows the explicit setting of the mobility",
    "2 Integers, 1 Number or 1 Integer, 1 Table: Point and site indices and a mobility value: 0 for fixed, 1 for free. If a site is not provided then a table can be passed to set all site at the point.",
    "",
    function (mep,p,s,v)
	if type(s) == type(1) then
	    mep:_setImageSiteMobility(p,s,v)
	    return
	end
	if type(s) == type({}) then
	    for i=1,mep:numberOfSites() do
		mep:setPointSiteMobility(p,i,s[i])
	    end
	    return
	end
	error("Invalid arguments")
    end
}

methods["pointSiteMobility"] =
    {
    "Usually the endpoints are fixed for the MEP algorithm. This allows the explicit retreival of the mobility",
    "1 or 2 Integers: Point and site indices. If the site index is not provided then a table of mobilities will be returned",
    "1 Number or 1 Table:  Mobility value: 0 for fixed, 1 for free. If the site is not specified then a table of site mobilities at the given point will be returned",
    function (mep,p,s)
	if s == nil then
	    local t = {}
	    for i=1,mep:numberOfSites() do
		t[i] = mep:pointSiteMobility(p,i)
	    end
	    return t
	end
	return mep:_getImageSiteMobility(p,s)
    end
}

methods["movePoint"] = 
    {
    "Change the coordinates of a point by the given amounts",
    "1 Integer, 1 Table of 3*:numberOfSites() Numbers, 1 optional Number: Point index, coordinate changes, scale factor (default 1)",
    "",
    function(mep, p, delta, scale)
	scale = scale or 1
	for s=1,mep:numberOfSites() do
	    local x1,y1,z1,c = mep:_nativeSpin(p,s)

	    local q = (s-1)*3
	    local x2 = x1 + scale * delta[q+1]
	    local y2 = y1 + scale * delta[q+2]
	    local z2 = z1 + scale * delta[q+3]
	    --print(p,s,x1,y1,z1, x2,y2,z2, c)
	    mep:setSpinInCoordinateSystem(p,s,x2,y2,z2,c)
	end
    end
}

methods["setSpinInCoordinateSystem"] =
    {
    "Set site as current coordinate system or given coordinate system",
    "2 Integers, 3 Numbers, 1 Optional String: 1st integer is path index, 2nd integer is site index, 3 numbers represent the new vector in the current coordinate system. If a string is given then the input is assumed to be in that coordinate system, otherwise the current coordinate system is used.",
    "",
    function(mep, ...)
	local cs = mep:coordinateSystem()
	local nums = {}
	local txts = {}
	local tabs = {}

	for i=1,table.maxn(arg) do
	    if type(arg[i]) == type({}) then
		error("setSpinInCoordinateSystem no longer accepts tables")
	    end
	    if type(arg[i]) == type(1) then
		table.insert(nums, arg[i])
	    end
	    if type(arg[i]) == type("s") then
		table.insert(txts, arg[i])
	    end
	end

	local p, s = nums[1] or 0, nums[2] or 0 -- idx
	
	table.remove(nums, 1)
	table.remove(nums, 1)

	if txts[1] == nil then
	    txts[1] = mep:coordinateSystem(p,s)
	end

	local x,y,z,c = nums[1], nums[2], nums[3], txts[1]

	-- print("SCS", x,y,z,c)
	x,y,z,c = mep:convertCoordinateSystem(x,y,z,c,"Cartesian")

	mep:setSpin(p,s, x,y,z,c)
    end
}


local tcopy = nil
function tcopy(t)
    if type(t) == type({}) then
	local c = {}
	for k,v in pairs(t) do
	    c[ tcopy(k) ] = tcopy(v)
	end
	return c
    end
    if type(t) == type("a") then
	return t .. ""
    end
    return t
end

local function ors_build(base, current, n, pop)
    if n == 0 then
	table.insert(base, current)
	return
    end

    for k,v in pairs(pop) do
	local c = tcopy(current)
	table.insert(c, v)
	ors_build(base, c, n-1, pop)
    end
end


methods["findBestPath"] =
{
"Find best path. JSON = {orientationsPerSite=n(default 20)}}",
"Optional JSON style table",
"",
function(_mep, JSON)
    JSON = JSON or {}
    local orientationsPerSite = JSON.orientationsPerSite or 20


    if _mep:numberOfPoints() ~= 2 then
	error("Require exactly 2 points in mep object.")
    end

    local mep = _mep:copy()

    local ns = mep:numberOfSites()

    local ors = math.vectorIcosphere(orientationsPerSite)

    local ors_combinations = {}

    ors_build(ors_combinations, {}, ns, ors)

    mep:setInitialPath(ors_combinations)

    local pe = mep:pathEnergy()
    
    mep:_findBestPath(ors_combinations, pe, _mep:pointInCoordinateSystem(1, "Cartesian"), _mep:pointInCoordinateSystem(2, "Cartesian"))

end
}
methods["findBestPath"] = nil -- disabling until after ising-like approx 




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


