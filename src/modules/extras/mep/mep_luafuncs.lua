-- MEP
local MODNAME = "MEP"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time
local methods = {}


local type_number = type(1)
local type_table = type({})
local type_function = type(type)
local type_text = type("")
local type_nil = type(nil)

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



local function getStepMod(tol, tableDiff)
    local step = {}
    local good = {}
    for k,err in pairs(tableDiff) do
        if err == 0 then
            good[k] = true
            step[k] = 1.01
        else
            step[k] = (tol/err) ^ (0.5)
            good[k] = step[k] >= 1

            if step[k] < 1 then
                step[k] = 0.75 * step[k]
            end
            --[[
            if err<tol then
                local x = 0.9*(tol / err)^(0.5)
                good[k] = true
                step[k] = x
            else
                local x = 0.95 * (tol / err)^(0.9)
                good[k] = false
                step[k] = x
            end
            --]]
        end
    end

    return step, good
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
	local ss = mep:spinSystem()
	
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
	mep:calculateEnergies()
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

methods["reportCustomConfiguration"] = 
{
    "Use a given report function to report a configuration",
    "1 Function, 1 Configuration ,1 Optional String: Report function, configuration, optional string prefix",
    "",
    function(mep, report, cfg, prefix)
        prefix = prefix or ""

        for s,c in pairs(cfg) do
            -- using cartesian system to move values to traditional ranges:
            local a1,a2,a3,a4 = c[1],c[2],c[3],c[4] or "Cartesian"
            local b1,b2,b3,b4 = mep:convertCoordinateSystem(a1,a2,a3,a4, "Cartesian")
            local c1,c2,c3,c4 = mep:convertCoordinateSystem(b1,b2,b3,b4, a4)
            report(string.format("%s%2d   % 6e  % 6e  % 6e  %s", prefix,s,c1,c2,c3,c4))
        end
    end
}

methods["reportConfigurationAtPoint"] =
    {
    "Use a given report function to report the configuration at a point",
    "1 Function, 1 Integer: Report function, path point index",
    "",
    function(mep, report, idx)
        mep:reportCustomConfiguration(report, mep:point(idx), string.format("%3d   ", idx))
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
    "0 or 1 Integers, 1 Optional Table of Booleans: Current or new number of path points. The table will have flags for each site defining if the interpolation should go around the back or long side of the sphere.",
    "",
    function(mep, _np, backward)
	local np = _np or mep:numberOfPathPoints()
	mep:resampleStateXYZPath(np, backward)
	-- mep:setNumberOfPathPoints(np)
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

        local type_v1 = type(v1)
        local type_v2 = type(v2)

	if type_v1 ~= type_v2 then
	    error("first two arguments must be the same type")
	end

	if v1 == nil then
	    error("nil input")
	end

	if type_v1 == type_number then -- integers
	    for s=1,mep:numberOfSites() do
		if not mep:equal({v1,s}, {v2,s}, tol) then
		    return false
		end
	    end
	    return true
	end

	if type_v1 == type_table then
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


local function single_compute_step(mep)
    local d = get_mep_data(mep)

    mep:initialize()

    mep:calculateEnergyGradients()
    --mep:makeForcePerpendicularToPath()
    --mep:makeForcePerpendicularToSpins()
    mep:applyForces()
end

local function filter_arg(arg)
    local result = {}
    result[ type_number ] = {}
    result[ type_text ] = {}
    result[ type_table ] = {}
    result[ type_function ] = {}
    for i=1,table.maxn(arg) do
        local type_argi = type(arg[i])
	result[ type_argi ] = result[ type_argi ] or {}
	table.insert(result[ type_argi ], arg[i])
    end
    return result
end

methods["execute"] =
    {
    "Run several relaxation steps of the Minimum Energy Pathway method with the ability to run until goals are met using constant steps and tolerances or a list of step counts and tolerances.",
    "1 Optional JSON-style table: Key \"steps\" is an integer or table of integers and give the number of steps to run, default 10. Key \"tolerance\" can be a single value or a table of values, default mep:tolerance(). Key \"report\" defines a print function, default nil. Key \"resamplePath\" with a boolean value, default true. Key \"goal\" is a number or a table of numbers and defines a goal average change in energy barrier after a given number of steps, if the goal is not met the MEP will loop, this can result in an infinite loop, default nil. Key \"maxIterations\" to limit the total number of iterations, default math.huge .",
    "",
    function(mep, json)
        json = json or {}
	local n    = json.steps or 10
	local tol  = json.tolerance or mep:tolerance()
        local _do_resample = json.resamplePath or true
	local report = json.report or function() end
	local goal = json.goal or nil
        local maxIteration = json.maxIterations or math.huge
        local quitOnMultiMax = false
        if json.quitOnMultiMax ~= nil then
            quitOnMultiMax = json.quitOnMultiMax
        end

	local d = get_mep_data(mep)
	local ss = mep:spinSystem()
	local np = mep:numberOfPathPoints()
        local ns = mep:numberOfSites()
	local energy_function = d.energy_function
	local currentIteration = 0
        local step_mod, good_step

	if ss == nil then
	    error("SpinSystem is nil. Set a working SpinSystem with :setSpinSystem")
	end
	
	if energy_function == nil then
	    error("Energy function is nil. Set an energy function with :setEnergyFunction")
	end
	
	mep:initialize()

	local function run_round(n, tol)
            report("Running " .. n .. " steps at tol = " .. tol)

            --for i = 1,n do
            local successful_steps = 0
            while successful_steps < n and currentIteration < maxIteration do
                currentIteration = currentIteration + 1
                local current_beta = mep:beta()
                local last_successful_steps = successful_steps
                
                -- first bool below is for "do_big" second is for "do_small"
                copy_to_children(mep, true, tol>=0)
                
                d.big_step:setBeta(current_beta)
                single_compute_step(d.big_step)
                
                if tol > 0 then -- negative tolerance will indicate no adaptive steps
                    d.small_step:setBeta(current_beta/2)
                    
                    single_compute_step(d.small_step)
                    single_compute_step(d.small_step)
                    
                    local totalDiff, tableDiff = d.big_step:absoluteDifference(d.small_step)
                    local aDiffAvrg = totalDiff / np
                    
                    step_mod, good_step = getStepMod(tol, tableDiff)

                    local smallest_step = step_mod[1]
                    local biggest_step = step_mod[1]

                    for p,good in pairs(good_step) do
                        if step_mod[p] < smallest_step then
                            smallest_step = step_mod[p]
                        end
                        if step_mod[p] > biggest_step then
                            biggest_step = step_mod[p]
                        end
                    end
                    
                    if smallest_step > 1 then
                        d.small_step:internalCopyTo(mep)
                        successful_steps = successful_steps + 1
                        mep:setBeta(biggest_step * current_beta)
                    else
                        mep:setBeta(smallest_step * current_beta)
                    end
                else -- negative or zero tolerance: no adaptive step
                    successful_steps = successful_steps + 1
                    d.big_step:internalCopyTo(mep)
                end
                
                -- interactive()
                mep:resamplePath(np)
                
                if _do_resample then
                    if last_successful_steps ~= successful_steps then
                        mep:resamplePath(np)
                    end
                end
            end
            
            return successful_steps
        end -- run_round

        local function getat(a, i)
            if type(a) == type_table then
                return a[i]
            end
            return a
        end
        local function max(a,b)
            if a>b then return a end
            return b
        end
        local function size(t)
            if type(t) == type_table then
                return table.maxn(t)
            end
            return 1
        end

        local function _cpe() -- critical point energies
            local path = mep:path()
            mep:reduceToInterpolatedCriticalPoints()
            local cpe = mep:pathEnergy()
            mep:setInitialPath(path)
            return cpe
        end

        local function sameCPE(a,b,tol)
            local na, nb = table.maxn(a), table.maxn(b)
            if na ~= nb then
                return false, 0, -1
            end

            local diffs = {}

            --[[
            report("Comparing:")
            report(table.concat(a, "\t"))
            report(table.concat(b, "\t"))
            --]]

            for i=1,na do
                diffs[i] = math.abs(a[i] - b[i])
            end

            local maxi = 1
            for i=2,na do
                if diffs[i] > diffs[maxi] then
                    maxi = i
                end
            end

            report(diffs[maxi], tol)
            return diffs[maxi] <= tol, diffs[maxi], maxi
        end
            

        if goal == nil then
            local m = max(size(n), size(tol))
            for i=1,m do
                if quitOnMultiMax then
                    if mep:maxCount() > 1 then
                        return
                    end
                end
                local _t = getat(tol, i)
                local _n = getat(n, i)
                if _t and _n then
                    run_round(_n, _t)
                end
            end
        else
            local old_cpe = _cpe()
            for g=1,size(goal) do
                local over_goal = true
                while over_goal and currentIteration < maxIteration do
                    if quitOnMultiMax then
                        if mep:maxCount() > 1 then
                            return
                        end
                    end
                    local _t = getat(tol, g)
                    local _n = getat(n, g)
                    if _t and _n then
                        run_round(_n, _t)
                        local new_cpe = _cpe()
                        local target = getat(goal, g) * _n
                        local same, off, idx = sameCPE(old_cpe, new_cpe, target)
                        --off = off * mep:beta()
                        if idx < 0 then
                            report("Critical point count mismatch")
                        else
                            report(string.format("target(%6e) diff(%6e) iteration(%d)", target, off, currentIteration))
                        end

                        if same then
                            over_goal = false
                        end
                        old_cpe = new_cpe
                    else
                        over_goal = false
                    end
                end
            end
            
        end
    end
}



methods["setSpinSystem"] =
    {
    "Set the SpinSystem that will be used to do energy and orientation calculations. Any changes made to this SpinSystem will be undone before control is returned to the calling environment.",
    "1 *SpinSystem*: SpinSystem to be used in calculations",
    "",
    function(mep, ss)
	-- local d = get_mep_data(mep)
	if getmetatable(ss) ~= SpinSystem.metatable() then
	    error("setSpinSystem requires a SpinSystem", 2)
	end
        mep:_setSpinSystem(ss)
	-- d.ss = ss
    end
}


methods["spinSystem"] =
    {
    "Get the *SpinSystem* used in the calculation",
    "",
    "1 SpinSystem",
    function(mep)
        return mep:_getSpinSystem()
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
	    mep:setSpinInCoordinateSystem(p,s,x1,x2,x3,c)
	end
    end
}

-- need sorted input
local function optCurveScore(x, goal)
    local score = 0
    local dscore = 0
    local matches = 0
    for k,v in pairs(x) do
        if goal[k] == 0 then
            dscore = - math.abs(x[k])
        else
            dscore = goal[k] * x[k]
            if dscore > 0 then
                matches = matches + 1
            end
        end
        -- print(goal[k], x[k], dscore)
        score = score + dscore
    end
    return score, matches
end

local function makeBasis(i, ns)
    local b = {}
    for j=1,ns do
        b[j] = {0,0,0}
    end

    local k = 1
    while i > 3 do
        k = k + 1
        i = i - 3
    end

    b[k][i] = 1
    return b
end

local function scaleAddCfg(cfg, delta, scale)
    local result = {}
    for i=1,table.maxn(cfg) do
        result[i] = {}
        result[i][1] = cfg[i][1] + delta[i][1] * scale
        result[i][2] = cfg[i][2] + delta[i][2] * scale
        result[i][3] = cfg[i][3] + delta[i][3] * scale
        result[i][4] = cfg[i][4]
    end
    return result
end

methods["optimizeCurvatureAtPoint"] =
    {
    "Move the given point so that the eigenvalues of the hessian match the (unordered) signs in the goal. Example:"..
        "<pre>mep:optimizeCurvatureAtPoint(1, {-1, 0, 0, 0, 1, 1})</pre>",
    "1 Integer, 1 Table of Numbers, Optional JSON options: Point Index, signs of the terms in the eigenvalues. JSON options: \"basis\" = table of tables of tables of numbers representing directions to consider, example: b1 = {{1,0,0}, {0,0,0}} b2 = {{0,1,0}, {0,0,0}} ... basis = {b1,b2,...} (default unit vectors aligned with each coordinate). \"scale\" = Table of Numbers, initial scale for each basis vector (default Table of 1e-3). \"steps\" = number of iterations (default 10). \"report\" = reporting function (default nil)",
    "",
    function(mep, p, _goal, JSON)
        JSON = JSON or {}
        local steps = JSON.steps or 10
        local basis = JSON.basis
        local report = JSON.report or function()end
        local skip = JSON.skip or 0
        if basis == nil then
            basis = {}
            for i=1,mep:numberOfSites()*3 do
                basis[i] = makeBasis(i, mep:numberOfSites())
            end
        end

        local scale = JSON.scale
        if scale == nil then
            scale = {}
        end
        for i=1,table.maxn(basis) do
            scale[i] = scale[i] or 1e-3
        end

        local goal = {}
        for k,v in pairs(_goal) do
            goal[k] = v
        end
        table.sort(goal)


        local function score(pt)
            local hessian = mep:hessianAtCustomConfiguration(pt)

            local cuts = {}
            for i=1,mep:numberOfSites() do
                cuts[i] = (i-1)*3+1
            end
            -- for 2 sites cuts = {1,}

            -- assuming spherical or canonical
            hessian =  hessian:cutX(cuts):cutY(cuts)

            local evals, evecs = hessian:matEigen()

            local negs = 0

            local ev = {}
            for i=1,evals:nx() do
                ev[i] = evals:get(i)
            end

            table.sort(ev, function(a,b) return math.abs(a) < math.abs(b) end)
            local prod = 1
            for i=1+skip,table.maxn(ev) do
                if ev[i] < 0 then
                    negs = negs + 1
                end
                prod = prod * math.abs(ev[i])
            end

            if negs ~= 1 then
                prod = -prod
            end

            return prod, evals, evecs
        end

        local best_pt = mep:point(p)
        local best_score = score(best_pt)

        for i=1,steps do
            for b=1,table.maxn(basis) do
                local consider_pt = scaleAddCfg(best_pt, basis[b], scale[b])
                local consider_score, evals, evecs = score(consider_pt)

                if consider_score > best_score then
                    scale[b] = scale[b] * 2
                    best_score = consider_score
                    best_pt = consider_pt
                    print(best_score)
                    evals:matPrint("evals")
                    evecs:matPrint("evecs")
                    for q=1,2 do
                        print(table.concat(best_pt[q], "\t"))
                    end
                    print()

                    --[[
                    local path = mep:path()
                    table.insert(path, best_pt)
                    mep:setInitialPath(path)
                    --]]
                else
                    scale[b] = -0.75 * scale[b]
                end
            end
        end

        mep:setPoint(p, best_pt)
        
        -- interactive()

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
    "N Integers or 1 Table of integers: The points to keep while deleting the others.",
    "",
    function(mep, ...)
	if type(arg[1]) == type_table then -- we were given a table
	    arg = arg[1]
	end

	local keepers = arg
	table.sort(keepers)
	
	local new_path = {}
	for i,p in ipairs(keepers) do
            new_path[i] = mep:interpolatePoint(p)
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
	if type(arg[1]) == type_table then -- we were given a table
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
	if type(a) == type_number then
	    a = {a}
	end
	if type(b) == type_number then
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
	if type(a) == type_number then
	    a = {a}
	end
	if type(b) == type_number then
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

	if type(a) == type_number then
	    a = {a}
	end
	if type(b) == type_number then
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
	if type(func) ~= type_function then
	    error("setEnergyFunction requires a function", 2)
	end
	d.energy_function = func
        mep:_setEnergyFunction(func)
    end
}


methods["setSites"] = 
    {
    "Set the sites that are allowed to move to transition from the initial configuration to the final configuration.",
    "1 Table of 1,2 or 3 Component Tables: mobile sites.",
    "",
    function(mep, tt)
	if type(tt) ~= type_table then
	    error("setSites requires a table of 1,2 or 3 Component Tables representing sites.") 
	end
	
	if tt[1] and type(tt[1]) ~= type_table then
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
	
	local msg = "setInitialPath requires a Table of Tables of site orientations."
	if type(pp) ~= type_table then
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
	    
	    if type(pp[p]) ~= type_table then
		error(msg .. " type(input[" .. p .. "]) = " .. type(pp[p]))
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
		mep:_addStateXYZ({x,y,z,c})
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
	
	return mep:_pathEnergyNDeriv(n)
    end
}

local function relaxPoint(mep, pointNum, steps, goal)
    local mep_data = get_mep_data(mep)
    if mep_data.isInitialized == nil then
	mep:initialize()
    end
    local energy_function = mep:energyFunction()
    local tol = nonDefaultTol or mep:tolerance()
    
    steps = steps or 10
    goal = goal or 0
    local h, good_steps = nil
    for i=1,steps do
	min_mag_grad, h, good_steps = mep:_relaxSinglePoint_sd(pointNum, h, goal)
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

methods["energyMinimizationAtPoint"] = 
    {
    "Attempt to minimize energy at a point.",
    "1 Integer, 1 Optional Integer, 1 Optional Number: Point index, number of steps (default 50), starting step size (default 1e-3)",
    "1 Integer, 3 Number: Ratio of successful steps to total steps, initial energy, final energy, final step size",
    function(mep, point, steps, h)
	steps = steps or 50

	local good_steps, iE, fE, fH = mep:_expensiveEnergyMinimization(point, (h or 1e-3), steps)

	return good_steps / steps, iE, fE, fH
    end
}

methods["energyMaximizationAtPoint"] = 
    {
    "Attempt to maximize energy at a point.",
    "1 Integer, 1 Optional Integer, 1 Optional Number: Point index, number of steps (default 50), starting step size (default 1e-3)",
    "1 Integer, 3 Number: Ratio of successful steps to total steps, initial energy, final energy, final step size",
    function(mep, point, steps, h)
	steps = steps or 50

	local good_steps, iE, fE, fH = mep:_expensiveEnergyMaximization(point, (h or 1e-3), steps)

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
	mep:energyMinimizationAtPoint(1, steps, h)
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
    "2 Tables of Vectors, 1 Number, 1 Optional Boolean: Each Vector is defined as 3 numbers and 1 optional string stating the coordinate system (Default Cartesian). The number is a ratio ideally between 0 and 1 which interpolates between the two input. The last optional boolean, if true, will interpolate the point around the back or long side of the sphere.",
    "1 Table of Vectors: The interpolated point.",
    function(mep, p1, p2, r, back)
	return mep:_interpolateBetweenCustomPoints(p1, p2, r, back)
    end
}



local phi = (1 + math.sqrt(5)) / 2
local resphi = 2 - phi

local goldenSectionSearch = nil
local function goldenSectionSearch(a,b,c, fa,fb,fc, tau, f, n)
    local x
    if (c - b) > (b - a) then
	x = b + resphi * (c - b)
    else
	x = b - resphi * (b - a)
    end

    if (math.abs(c - a) < tau * (math.abs(b) + math.abs(x))) or n <= 0 then
	-- return (c + a) / 2 -- this return misses minimums at boundaries
        if c < a then
            return c
        end
        return a
    end

    local fx = f(x)
    
    --assert(f(x) != f(b));
    if (fx < fb) then 
	if (c - b > b - a) then
	    return goldenSectionSearch(b,x,c, fb,fx,fc, tau, f, n-1)
	else 
	    return goldenSectionSearch(a,x,b, fa,fx,fb, tau, f, n-1)
	end
    else 
	if (c - b > b - a) then
	    return goldenSectionSearch(a,b,x, fa,fb,fx, tau, f, n-1)
	else
	    return goldenSectionSearch(x,b,c, fx,fb,fc, tau, f, n-1)
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
"1 Optional JSON style table: Keys are: \"Tolerance\", tau as defined in the golden search wikipedia page (Default 1e-8), \"InterpolateEndPoints\", boolean to interpolate end points (default true). \"MaxIterations\", default 16. JSON table can also contain the keys \"addMinPadding\", \"addMaxPadding\" and \"addPadding\" with values of numbers. These values are interpreted as ratios of the distance between a point and it's previous and next critical point. These distances will determine how close padding points will be added encompasing minimums, maximums or all interpolated critical points. Padding is not added before the first point or after the last point. The value of the \"addPadding\" key overides the other Paddings.",
"4 Tables of Numbers: Tables of non-integer point along path corresponding to each interpolated set of minimum points, maximum points ad all points. These Non-integer points can be transformed into points with the :interpolatePoint() method. The last table of ordered numbers includes the original points minus the integer points closest to the interpolated points.",
function(mep, json)
    json = json or {}
    local tol = json.Tolerance or 1e-8
    local endpts = json.InterpolateEndPoints or true
    local maxi = json.MaxIterations or 16

    local mins, maxs, all = mep:criticalPoints()
    local imins, imaxs, iall = {}, {}, {}

    local function min_func(x)
	return mep:energyOfCustomConfiguration( mep:interpolatePoint(x ))
    end
    local function max_func(x)
	return -min_func(x)
    end
    local tau = tol
    local np = mep:numberOfPoints()

    for i,v in pairs(mins) do
        if (v == 1 or v == np) and (endpts == false) then
            imins[i] = v
        else
            local x1,x3 = boundingMin(mep, v)
            local f = min_func

            local fx1, fx3 = f(x1), f(x3)
            local x2 = (x1+x3)/2
            local x2 = goldenSectionSearch(x1,x2,x3, fx1,f(x2),fx3, tau, f, maxi)
            local fx2 = f(x2)
            if fx1 < fx2 then
                x2 = x1
                fx2 = fx1
            end
            if fx3 < fx2 then
                x2 = x3
                fx2 = fx3
            end
            
            imins[i] = x2
        end
    end

    for i,v in pairs(maxs) do
        if (v == 1 or v == np) and (endpts == false) then
            imaxs[i] = v
        else
            local x1,x3 = boundingMax(mep, v)
            local f = max_func
            local fx1, fx3 = f(x1), f(x3)
            local x2 = (x1+x3)/2
            local x2 = goldenSectionSearch(x1,x2,x3, fx1,f(x2),fx3, tau, f, maxi)
            local fx2 = f(x2)

            if fx1 < fx2 then -- these "less than"s are right, max_func negates values
                x2 = x1
                fx2 = fx1
            end
            if fx3 < fx2 then
                x2 = x3
                fx2 = fx3
            end
            
            imaxs[i] = x2
        end
    end


    for k,v in pairs(imins) do
        table.insert(iall, v)
    end

    for k,v in pairs(imaxs) do
        table.insert(iall, v)
    end


    table.sort(iall)
    local imins_n = table.maxn(imins)
    local imaxs_n = table.maxn(imaxs)
    local iall_n = table.maxn(iall)

    local function before_dist(value)
        for i=2,iall_n do
            if iall[i] == value then
                return iall[i] - iall[i-1]
            end
        end
    end
    local function after_dist(value)
        for i=1,iall_n-1 do
            if iall[i] == value then
                return iall[i+1] - iall[i]
            end
        end
    end

    if json.addPadding then
        json.addMinPadding = json.addPadding
        json.addMaxPadding = json.addPadding
    end

    if json.addMinPadding then 
        local n = table.maxn(imins)
        local r = json.addMinPadding
        for i=1,n do
            local d = before_dist(imins[i])
            if d then
                table.insert(imins, imins[i] - d*r)
                table.insert(iall, imins[i] - d*r)
            end

            d = after_dist(imins[i])
            if d then
                table.insert(imins, imins[i] + d*r)
                table.insert(iall, imins[i] + d*r)
            end
        end
    end

    if json.addMaxPadding then 
        local n = table.maxn(imaxs)
        local r = json.addMaxPadding
        for i=1,n do
            local d = before_dist(imaxs[i])
            if d then
                table.insert(imaxs, imaxs[i] - d*r)
                table.insert(iall, imaxs[i] - d*r)
            end

            d = after_dist(imaxs[i])
            if d then
                table.insert(imaxs, imaxs[i] + d*r)
                table.insert(iall, imaxs[i] + d*r)
            end
        end
    end

    table.sort(imins)
    table.sort(imaxs)
    table.sort(iall)


    local icomplete = {}
    
    for i=1,mep:numberOfPoints() do
        icomplete[i] = i
    end

    local function round(x)
        local decimal = math.fmod(x, 1)
        if decimal > 0.5 then
            return math.floor(x) + 1
        else
            return math.floor(x)
        end
    end

    for k,v in pairs(iall) do
        local r = round(v)
        icomplete[r] = v
    end

    return imins, imaxs, iall, icomplete
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

    local icp = mep:interpolateBetweenCustomPoints(cfg1, cfg2, r)
    return icp
end
}


methods["mainVectorBetweenPoints"] =
    {
    "Return the most dominant Cartesian vector between two sets of points. This is done by combuting the cartesian difference between each pair of elements in the points and selecting the longest vector.",
    "2 Integers: Indices in calling MEP.",
    "1 Vector: The largest difference between the points.",
    function(mep,p1,p2)
        return mep:mainVectorBetweenCustomPoints(
            mep:point(p1),
            mep:point(p2))
    end
}

methods["mainVectorBetweenCustomPoints"] =
    {
    "Return the most dominant Cartesian vector between two sets of points. This is done by combuting the cartesian difference between each pair of elements in the points and selecting the longest vector.",
    "2 Tables of Vectors: Each Vector is defined as 3 numbers and 1 optional string stating the coordinate system (Default Cartesian).",
    "1 Vector: The largest difference between the points.",
    function(mep, p1, p2)
        if table.maxn(p1) ~= table.maxn(p2) then
            error("points are not comprised of same number of vectors")
        end
        local d = {}
        for i=1,table.maxn(p1) do

            local x1,x2,x3 = mep:convertCoordinateSystem(p1[i][1],p1[i][2],p1[i][3],p1[i][4] or "Cartesian", "Cartesian")
            local y1,y2,y3 = mep:convertCoordinateSystem(p2[i][1],p2[i][2],p2[i][3],p2[i][4] or "Cartesian", "Cartesian")
            d[i] = {y1-x1, y2-x2, y3-x3}
        end

        local function n2(x)
            return x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
        end

        local maxi = 1
        local maxv = n2(d[1])

        for i=2,table.maxn(p1) do
            local herev = n2(d[i])
            if herev > maxv then
                maxv = herev
                maxi = i
            end
        end

        return {d[maxi][1], d[maxi][2], d[maxi][3], "Cartesian"}
    end
}

methods["rotatePathAboutBy"] =
    {
    "Rotate all vectors in the path about the given vector by the given number of radians",
    "1 Vector, 1 Number: Vector is 1 table of 3 numbers and 1 optional string or 3 numbers and 1 optional string. Numbers repesent coordinates, string is coordinate system (default Cartesian). Last number is the angle to rotate about by (radians)",
    "",
    function(mep, a,b,c,d,e)
        if type(a) == type_table then
            return mep:_rotatePathAboutBy(a[1], a[2], a[3], a[4] or "Cartesian", b)
        end

        if type(d) == type_text then
            return mep:_rotatePathAboutBy(a,b,c,d,e)
        end

        return mep:_rotatePathAboutBy(a,b,c,"Cartesian", d)
    end
}

methods["gradientMinimizationAtPoint"] = 
    {
    "Attempt to minimize the energy gradient at a point.",
    "1 Integer, 1 Optional Integer, 1 Optional Number: Point index, number of steps (default 50), starting step size (default 1e-3)",
    "1 Integer, 3 Number: Ratio of successful steps to total steps, initial square of the gradient, final square of the gradient, final step size",
    function(mep, point, steps, h)
        steps = steps or 50

        local good_steps, iG, fG, fH = mep:_expensiveGradientMinimization(point, (h or 1e-3), steps)

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
	
	local numSteps = numSteps or 50
	
	if pointNum == nil then
	    error("Require point number")
	end
	
	if direction == nil then
	    direction = -1
	end
	
	return mep:_slidePoint(pointNum, direction, numSteps, stepSize)
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

	local t1, t2, t3 = mep:_classifyPoint(pointNum, step)
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
	local t = mep:_gradAtPoint(pointNum)
	
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
	if c1 < 1 or c1 > mep:numberOfSites()*3 then
	    error("c1 out of bounds")
	end

	if c2 < 1 or c2 > mep:numberOfSites()*3 then
	    error("c2 out of bounds")
	end

	if pointNum < 1 or pointNum > mep:numberOfPathPoints() then
	    error("point index out of bounds")
	end

	local t = mep:_computePointSecondDerivativeAB(pointNum, c1, c2, h1, h2)
	return t
    end

}

methods["hessianAtPoint"] = {
    "Compute the 2nd order partial derivative at a given point along the path.",
    "1 Integer, 1 Optional Array: Point to calculate 2nd derivative about. If no array is provided one will be created",
    "1 Array: Derivatives",
    function(mep, pointNum, destArray)
	
	local c = mep:siteCount()
	
	if destArray then
	    if destArray:nx() ~= c*3 or destArray:ny() ~= c*3 then
		error("Destination array size mismatch. Expected " .. c*3 .. "x" .. c*3)
	    end
	else
	    destArray = Array.Double.new(c*3, c*3)
	end
	
        --local ep = mep:epsilon()
        --mep:setEpsilon(ep*0.01)
	local t = mep:_hessianAtPoint(pointNum)
        --mep:setEpsilon(ep)
	
	for x=0,c*3-1 do
	    for y=0,c*3-1 do
		destArray:set(x+1, y+1, t[x+y*(c*3)+1])
	    end
	end		
	return destArray
    end
}

methods["hessianAtCustomConfiguration"] = {
    "Compute the 2nd order partial derivative at a given custom configuration.",
    "1 Table of Sites, 1 Optional Array: Point to calculate 2nd derivative about. If no array is provided one will be created",
    "1 Array: Derivatives",
    function(mep, cfg, res)
        if mep:numberOfPoints() == 0 then
            error("MEP must have at least 1 point (to be used as a buffer)")
        end
        local p = mep:point(1)

        mep:setPoint(1, cfg)
        res = mep:hessianAtPoint(1, res)

        mep:setPoint(1, p) -- restore

        return res
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
	if type(arg[1]) == type_table then -- we were given a table
	    arg = arg[1]
	end
	local new_initial_path = {}

	for i,v in pairs(arg) do
	    new_initial_path[i] = mep:interpolatePoint(v)
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

methods["reduceToInterpolatedCriticalPoints"] = 
    {
    "Reduce the path to the points returned from the :interpolateCriticalPoints() method. The points between the critical points will be discarded and the total number of points will be reduced to the number of critical points.",
    "1 optional JSON style table: same as JSON table to be passed to MEP:interpolatedCriticalPoints(). ",
    "",
    function(mep, json)
        json = json or {}
	local _, _, icps = mep:interpolatedCriticalPoints(json)
        -- > print(table.concat(icps, ","))   ABC
        -- 1.000000022465,6.3220078306327,10.999999916953

	mep:reduceToPoints(icps)
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
	
	if type(arg[1]) == type_table then
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
	
	if type(arg[1]) == type_table then
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
    "1 Table of 3 Numbers or 3 Numbers, 2 Strings: A vector given either as a table of numbers or as 3 numbers and Source and Destination Coordinate System names (must match one" ..
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
	    mep:energyMinimizationAtPoint(p, n, h)
	end
    end
}

methods["globalExcite"] =
    {
    "Run several excitation steps where each point takes steps to increase it's energy.",
    "1 Optional Integer, 1 Optional Number: Number of steps, default 50. Initial step size for each point.",
    "",
    function(mep, n, h)
	for p=1,mep:numberOfPoints() do
	    mep:energyMaximizationAtPoint(p, n, h)
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
	    if(type(v)) == type_number then -- number
		table.insert(nums, v)
	    end
	    if(type(v)) == type_table then -- table
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

local coordData = {}
coordData["x"] = {"Cartesian", 1}
coordData["y"] = {"Cartesian", 2}
coordData["z"] = {"Cartesian", 3}

coordData["r"] = {"Spherical", 1}
coordData["p"] = {"Spherical", 2}
coordData["t"] = {"Spherical", 3}

coordData["phi"] = {"Canonical", 2}

local function getPlotData(mep, what)
    local a,b,c,d = string.find(what, "(.*)(%d+)")
    local data = {}

    if a then
        local site = tonumber(d)
        if site < 1 or site > mep:numberOfSites() then
            error("Invalid site number")
        end

        local coord = string.lower(c)
        local cd = coordData[coord]

        for i=1,mep:numberOfPoints() do
            local a1,a2,a3 = mep:spinInCoordinateSystem(i, site, "Cartesian")
            local c1,c2,c3 = mep:convertCoordinateSystem(a1,a2,a3,"Cartesian", cd[1])
            local t = {c1,c2,c3}
            local v = t[cd[2]]

            data[i] = v
        end
        return data
    end

    if string.lower(what) == "energy" then
        return mep:pathEnergy()
    end

    if string.lower(what) == "number" then
        for i=1,mep:numberOfPoints() do
            data[i] = i
        end
        return data
    end

    error("unknown what:" .. what)
end

methods["plot"] =
    {
    "Use gnuplot to plot the path",
    [[Optional table of strings: Default {"Number", "Energy"}. Available: "xN", "yN", "zN", "rN", "pN", "tN", "phiN" where N is site number. "Number" for point number, "Energy" for point energy]],
    "",
    function(mep, what, json)
        local DISPLAY = os.getenv("DISPLAY") or ""
        what = what or {"Number", "Energy"}
        local datafile = os.tmpname()
        local cmdfile = os.tmpname()

        local f = io.open(datafile, "w")
        local pe = mep:pathEnergy()

        -- local function getPlotData(mep, what)

        local x = getPlotData(mep, what[1])
        local y = getPlotData(mep, what[2])

        for i=1,table.maxn(x) do
            f:write(string.format("%e %e\n", x[i], y[i]))
        end
        f:close()

        f = io.open(cmdfile, "w")

        if DISPLAY == "" then
            f:write("set term dumb\n")
        else
        end

        f:write("plot \"" .. datafile .. "\" w lp title \"\"\n")
        f:close()

        if DISPLAY == "" then
            os.execute("gnuplot " .. cmdfile)
        else
            os.execute("gnuplot -p " .. cmdfile .. " && sleep 1")
        end

        os.execute("rm -f " .. cmdfile .. " " .. datafile)

    end
}



methods["findMinimums"] =
    {
    "Find the coordinates of minima in the energy landscape",
    "1 Integer or input compatible with :setInitialPath(): Number of initial points which will be spread out evenly over hyper-sphere created by the sites involved in the calculation. If a table is instead given then the orientations in it will be used for the search start. Additionally, a JSON-style table can be provided as the second argument with data at a \"report\" key, data at a \"cs\" key and data at a \"hints\" key. The data at the report will be a function that will be called with human readable data regarding the progress of the algorithm (common choice is the print function). The cs key is the coordinate system used in the case that you are not specifying starting points but a point count and the program will evenly distribute them (a common choice, as well as the default, is \"Spherical\"). The hints data contains a list of starting points that should be added to whatever search will take place. ",
    "Table of minima: Each minimum will be a table of sites, each site is 3 numbers representing a vector and a string naming the coordinate system.",
    function(mep, n, json)
	json = json or {}
        local type_n = type(n)
	if type_n == type_number then -- need n points over hypersphere		
	    local cs = json.cs or "Spherical"
	    return mep:findMinimums(mep:evenlyDistributedPoints(n, cs), json)
	end
	if type_n == type_table then
	    local report = json.report or function() end
	    local hints = json.hints

	    mep:setInitialPath(n)

            -- mep:execute({steps=10, tolerance={0.75, 0.5, 0.1}, goal={1e-12, 1e-16, 1e-20}})
            -- ABC
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
		{      1,      1e-4,           60,     0.1},
		{      1,      1e-4,           60,     0.2},
		{      1,      1e-4,           60,     0.5},
 		{      3,      1e-4,           60,     0.3},
 		{      3,      1e-4,          100,     0.2}
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
		if j == 1 then
		    mid_up = num_up
		end
	    end
 
	    if mid_up ~= num_up then -- we may still be refining
		json.hints = nil -- clearing hints so this is just a continuation
		report("search has not converged, continuing search with current state")
		return mep:findMinimums(mep:path(), json)
	    end

	    -- finalization
	    for i=1,mep:numberOfPoints() do
		local ratio, step_size = 1, 1e-10
		local iteration = 0
		while ratio > 1/30 and iteration < 10 do
		    ratio, _, _, step_size = mep:energyMinimizationAtPoint(i, 30, step_size)
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
	

	error("`findMinimums' expects an integer or table as input")
    end
}

methods["findMaximums"] =
    {
    "Find the coordinates of maximums in the energy landscape",
    "1 Integer or input compatible with :setInitialPath(): Number of initial points which will be spread out evenly over hyper-sphere created by the sites involved in the calculation. If a table is instead given then the orientations in it will be used for the search start. Additionally, a JSON-style table can be provided as the second argument with data at a \"report\" key, data at a \"cs\" key and data at a \"hints\" key. The data at the report will be a function that will be called with human readable data regarding the progress of the algorithm (common choice is the print function). The cs key is the coordinate system used in the case that you are not specifying starting points but a point count and the program will evenly distribute them (a common choice, as well as the default, is \"Spherical\"). The hints data contains a list of starting points that should be added to whatever search will take place. ",
    "Table of maximums: Each maximum will be a table of sites, each site is 3 numbers representing a vector and a string naming the coordinate system.",
    function(mep, n, json)
	json = json or {}
        local type_n = type(n)
	if type_n == type_number then -- need n points over hypersphere		
	    local cs = json.cs or "Spherical"
	    return mep:findMaximums(mep:evenlyDistributedPoints(n, cs), json)
	end
	if type_n == type_table then
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
		{      1,      1e-4,           60,     1},
		{      1,      1e-4,           60,     2},
		{      1,      1e-2,           60,     5},
 		{      1,      1e-4,           60,     0.2},
 		{      1,      1e-6,           60,     1}
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
		    mep:globalExcite(rsteps, rtol)
		    local up = mep:uniquePoints(eqtol * math.pi/180)
		    num_up = table.maxn(up)

		    if table.maxn(up) == 0 then
			interactive("Zero unique points")
		    end
		    mep:reduceToPoints(up)
		    report(string.format("Global relax tolerance: %4g, " .. 
					 "Unique test degrees: %4g, Unique points: %4d",
				     rtol, eqtol, table.maxn(up)))
		end
		if j == 1 then
		    mid_up = num_up
		end
	    end
 
	    if mid_up ~= num_up then -- we may still be refining
		json.hints = nil -- clearing hints so this is just a continuation
		report("search has not converged, continuing search with current state")
		return mep:findMaximums(mep:path(), json)
	    end

	    -- finalization
	    for i=1,mep:numberOfPoints() do
		local ratio, step_size = 1, 1e-10
		local iteration = 0
		while ratio > 1/30 and iteration < 10 do
		    ratio, _, _, step_size = mep:energyMaximizationAtPoint(i, 30, step_size)
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
	

	error("`findMaximums' expects an integer or table as input")
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
            local type_argi = type(arg[i])
	    if type_argi == type_number then
		table.insert(ints, arg[i])
	    end
	    if type_argi == type_text then
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
        local type_s = type(s)
	if type_s == type_number then
	    mep:_setImageSiteMobility(p,s,v)
	    return
	end
	if type_s == type_table then
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
            local type_argi = type(arg[i])
	    if type_argi == type_table then
		error("setSpinInCoordinateSystem no longer accepts tables")
	    end
	    if type_argi == type_number then
		table.insert(nums, arg[i])
	    end
	    if type_argi == type_text then
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
    if type(t) == type_table then
	local c = {}
	for k,v in pairs(t) do
	    c[ tcopy(k) ] = tcopy(v)
	end
	return c
    end
    if type(t) == type_text then
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


--methods["dijkstrasPath"] =
methods["findPath"] =
{
"Run a modified Bellman-Ford algorithm over sets of subdivided dodecahedrons to find a coarse minimum energy path between a pair of points.",
"2 Tables, 1 optional Integer: Each table contains sites, each site is 3 numbers representing a vector and a string naming the coordinate system. The integer can be between 1 and 5 inclusively (default 1) and determines the number of subdivisions of the dodecahedron representing a sphere. Larger values will take a considerably longer amount of time to solve. A single site system with a subdivision count of 5 will result in approximately 750 vertices, each with 7 neighbours. A 2 site system with a subdivision count of 5 will result in half a million vertices, each with 49 neighbours. ",
"1 Table: compatible with :setInitialPath, represents the solved path from the start to the end.",
function(mep, a,b,n)
    local p = mep:_findBestPath(a,b,n)
    table.insert(p, 1, a)
    table.insert(p, b)
    return p
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


