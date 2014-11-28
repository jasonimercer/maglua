-- This file provides the function PathDynamics(mep) which, when given an MEP object containing only 1 maximum, will return a calculated description of the dynamics associated with the path in a table.  An optional second argument can be given to the function with a JSON style table. The following lists the keys that can be set for the JSON table:
--<dl>
--<dt>report</dt><dd>Function used to report internal progress of the algorithms, generally `print'</dd>
--<dt>onNan</dt> <dd>Function called when a nan is detected in the final stages of the calculation, generally `interactive' or `error'. The return value from this function will be assigned to the attempt frequency so we can recover when we hit bad cases.</dd>
--<dt>onMultipleHops</dt><dd>Function to call when multiple hops are detected</dd>
--<dt>EnergyMinimizationSteps</dt><dd>Number of steps to take to refine minima, default 100</dd>
--<dt>GradientMinimizationSteps</dt><dd>Number of steps to take to refine saddle point, default 100</dd>
--<dt>addResults</dt><dd>A table that will be joined to the return table</dd>
--</dl>
--This function is currently hard-coded to deal with points made of 2 sites. The return table is populated at the the following keys:
--<dl>
--<dt>EnergyBarrier</dt>    <dd>The energy difference between the saddle point and the start point</dd>
--<dt>AttemptFrequency</dt> <dd>The frequency at which the path will be attempted</dd>
--<dt>Event</dt>            <dd>A MagLua function that, when given a *SpinSsytem* will write the last point in the path.</dd>
--<dt>InitialEnergy</dt>    <dd>Energy at the start of the path</dd>
--<dt>MaximumEnergy</dt>    <dd>Energy at the maximum point of the path</dd>
--<dt>FinalEnergy</dt>      <dd>Energy at the end of the path</dd>
--<dt>Description</dt>      <dd>Terse description of the dynamics</dd>
--<dt>AtSingleWell</dt>           <dd>Boolean value set to true if system starts at the bottom of a well in a single well system</dd>
--<dt>ToSingleWell</dt>           <dd>Boolean value set to true if system represents a single well and the start point of the path is not at the bottom of the well</dd>
--<dt>MultipleHops</dt>     <dd>True if system has to transition over more than 1 maximum between the start and end inclusively</dd>
--<dt>Sites</dt>            <dd>The sites involved in the path</dd>
--<dt>InitialConfiguration</dt><dd>Table of Cartesian orientations at start of path</dd>
--<dt>FinalConfiguration</dt><dd>Table of Cartesian orientations at end of path</dd>
--<dt>MinimumConfigurations</dt><dd>A table of all configurations at minimum energy points</dd>
--<dt>txtSites</dt>            <dd>The sites involved in the path as a string</dd>
--<dt>txtInitialConfiguration</dt><dd>Table of Cartesian orientations at start of path converted to a string</dd>
--<dt>txtFinalConfiguration</dt><dd>Table of Cartesian orientations at end of path converted to a string</dd>
--</dl>
--
-- This file also provides the function GeneratePathDynamicsGraph(all_paths, MaxDegreesDifference, LowEnergy) which will convert a table of return values from PathDynamics into a table where each element represents a node in a graph. Each node is a table with elements:
--<dl>
--<dt>Configuration</dt><dd>Configuration matching one of the InitialConfigurations from the input paths</dd>
--<dt>Energy</dt><dd>Energy at node</dd>
--<dt>Edges</dt><dd>All the input paths with matching InitialConfigurations</dd>
--<dt>Low_Energy_Partner</dt><dd>A table of indexes of nodes who are considered low-energy transitions from the current nodes</dd>
--<dt>Cluster_Partner</dt><dd>A table of indexes of nodes who are mutually low-energy transition partners of the current node</dd>
--<dt>High_Energy_Partner</dt><dd>A table of indexes of nodes who are considered be not low-energy transitions from the current nodes</dd>
--<dt>Partner</dt><dd>A table of Low and High Partners</dd>
--</dl>
-- Furthermore, as a second return value from this function a search function will be returned that takes two node indexes and returns the corresponding all_path element from the all_paths table.
--
-- This file also provides the function AllPathDynamics(mep, minima, JSON) which will generate the all_paths input for the above function. The input is an MEP object, the return from mep:findMinima() and an optional JSON-style table with extra options. The keys that can be populated are:
--<dl>
--<dt>Report</dt><dd>Functon to use as the internal report function, default `function() end'. The print function could be used.</dd>
--<dt>SimplifyDegrees</dt><dd>Number of degrees to use for equality testing in the loop finding code, default 5.</dd>
--<dt>MultipleHopRetries</dt><dd>Number of times to attempt to re-refine a multiple hop path in the event that one of the extra minimum points is due to some noise. Default 2</dd>
--<dt>PathDynamicsOnNan</dt><dd>Function to provide the PathDynamics function's OnNan option, default `error'. If this function returns, it will be assigned to the attempt frequency. This allows us to give a best guess when the math goes sideways.</dd>
--<dt>PathDynamicsOnMultipleHops</dt><dd>Passed to PathDynamic's onMultipleHops</dd>
--<dt>PathDynamicsReport</dt><dd>Function to provide the PathDynamics function's Report option, default `function() end'.</dd>
--<dt>AllPathDynamicsMEP</dt><dd>Required function that solves an MEP between two states. Input is an MEP object, a starting configuration, an ending configuration and an optional reporting function. If this function is not provided then it's default value is the error function.</dd>
--</dl>


-- support functions:
local function txtPt(mep, p, cs)
    local txt = {}
    for s=1,mep:numberOfSites() do
	local x,y,z,c = mep:spinInCoordinateSystem(p,s,cs)
	txt[s] = "(" .. table.concat({x,y,z,"\""..c.."\""}, ", ") .. ")"
    end
    return table.concat(txt, ", ")
end

-- make a diagonal matrix
local function makeDiagonal(d)
    if type(d) ~= type({}) then
	d = d:toTable(1)
    end
    local n = table.maxn(d)
    local a = Array.Double.new(n,n)
    a:matMakeI()
    for i=1,n do
	a:set(i,i,d[i])
    end
    return a
end


function PathDynamics(mepOrig, json)
    local results = {}

    results["EnergyBarrier"] = 0
    results["AttemptFrequency"] = 0
    results["Event"] = function() end
    results["InitialEnergy"] = 0
    results["MaximumEnergy"] = 0
    results["FinalEnergy"] = 0
    results["Description"] = ""
    results["AtSingleWell"] = false
    results["ToSingleWell"] = false
    results["MultipleHops"] = false
    results["Sites"] = mepOrig:sites()
    results["MinimumConfigurations"] = {}

    json = json or {}
    local blank = function() end
    local addResults = json.addResults or {}
    local report = json.report or blank
    -- mpi root report functon
    local report1 = function(...) if mpi.get_rank() == 1 then report(...) end end
    local energyMinimizationSteps = json.EnergyMinimizationSteps or 100
    local gradientMinimizationSteps = json.GradientMinimizationSteps or 100
    local onMultipleHops = json.onMultipleHops or blank

    local mep = mepOrig:copy()
    local sites = mep:sites()

    mep:setCoordinateSystem("Canonical")
    mep:setTolerance(0.0001)

    -- interactive()

    -- getting important points
    local mins, maxs, all = mep:criticalPoints()

    -- add common results to the return value
    local function ar(res) -- add inputted results, add text
	for k,v in pairs(addResults) do
	    res[k] = v
	end

	local s = {}
	for k,v in pairs(results.Sites) do
	    s[k] = "{" .. table.concat(v, ", ") .. "}"
	end
	res.txtSites =  "{" .. table.concat(s, ", ") .. "}"

	local c = {}
	for k,v in pairs(results.InitialConfiguration) do
            c[k] = string.format("{%6.4e, %6.4e, %6.4e}", v[1], v[2], v[3])
	end
	res.txtInitialConfiguration =  "{" .. table.concat(c, ", ") .. "}"

	local c = {}
	for k,v in pairs(results.FinalConfiguration) do
            c[k] = string.format("{%6.4e, %6.4e, %6.4e}", v[1], v[2], v[3])
        end
	res.txtFinalConfiguration =  "{" .. table.concat(c, ", ") .. "}"

	return res
    end

    local function make_event_function(_sites, _cfg, _report1, _msg)
	-- make local copy for closure
	local ns = nil
	local sites = {}
	for k1,v1 in pairs(_sites) do
	    sites[k1] = {}
	    ns = k1
	    for k2,v2 in pairs(v1) do
		sites[k1][k2] = v2
	    end
	end
	local cfg = {}
	for k1,v1 in pairs(_cfg) do
	    cfg[k1] = {}
	    for k2,v2 in pairs(v1) do
		cfg[k1][k2] = v2
	    end
	end
	local msg = _msg
	local report1 = _report1

	local function event_function(ss)
	    if report1 and msg then
		report1(msg)
	    end
	    for s=1,ns do
		ss:setSpin(sites[s], cfg[s])
	    end
	end

	return event_function
    end


    results["InitialEnergy"] = mep:energyAtPoint(1)
    results["FinalEnergy"] = mep:energyAtPoint( mep:numberOfPoints())

    results["InitialConfiguration"] = {}
    for s=1,mep:numberOfSites() do
	local x,y,z = mep:spin(1,s)
	results["InitialConfiguration"][s] = {x,y,z}
    end

    results["FinalConfiguration"] = {}
    for s=1,mep:numberOfSites() do
	local x,y,z = mep:spin(mep:numberOfPoints(),s)
	results["FinalConfiguration"][s] = {x,y,z}
    end


    -- record all minimum configurations
    for k,idx in pairs(mins) do
	results.MinimumConfigurations[k] = mep:point(idx)
    end 


    if table.maxn(maxs) > 1 then -- it's a double or multi-hop transition. We don't deal with this
	onMultipleHops()
	results["MultipleHops"] = true
	results["Description"] = "Multiple hops, discarding case"
	report(results["Description"])
        return ar(results)
    end



    local txtSites = {}
    for k,v in pairs(sites) do
	txtSites[k] = "{" .. table.concat(v, ", ") .. "}"
    end
    txtSites = "{" .. table.concat(txtSites, ", ") .. "}"


    
    local txtDestination = txtPt(mep, mep:numberOfPoints(),"Cartesian")

    mep:reduceToPoints(all) -- reduce from N to 3 points
    local mins, maxs, all = mep:criticalPoints()


    -- making a local copy of sites and orientations
    -- so they can be used in closures without fear
    -- of external corruption
    local ns = mep:numberOfSites()
    
    local final = {}
    for s=1,ns do
	local x,y,z = mep:spin( mep:numberOfPoints(), s)
	final[s] = {x,y,z} -- cartesian, compatible with SpinSystem
    end


    -- this is newer code (May 06/2014) to cover the case of a single minima
    if table.maxn(all) == 2 then -- we are dealing with a single well system		
	if mep:energyAtPoint(all[1]) < mep:energyAtPoint(all[2]) then 
	    -- This case covers us at the bottom of a single well system
	    results["EnergyBarrier"] = math.huge
	    results["AttemptFrequency"] = 0
	    results["Event"] = function() end
	    results["InitialEnergy"] = mep:energyAtPoint(1)
	    results["MaximumEnergy"] = mep:energyAtPoint(1)
	    results["FinalEnergy"] = mep:energyAtPoint(1)
	    results["Description"] = string.format("%s is in a single well", txtSites)
	    results["AtSingleWell"] = true
	    results["ToSingleWell"] = true
	    results["MultipleHops"] = false

	    report(results["Description"])
	    return ar(results)
	else
	    results["EnergyBarrier"] = 0 -- fall down the well 
	    results["AttemptFrequency"] = 1e80 -- something absurdely large: spontaneous
	    results["Event"] = make_event_function(sites, final, report1, string.format( "spontaneously setting %s to %s", txtSites, txtDestination))
	    results["InitialEnergy"] = mep:energyAtPoint(1)
	    results["MaximumEnergy"] = mep:energyAtPoint(1)
	    results["FinalEnergy"] = mep:energyAtPoint(mep:numberOfPoints())
	    results["Description"] = string.format("%s set to spontaneously flip", txtSites)
	    results["AtSingleWell"] = false
	    results["ToSingleWell"] = true
	    results["MultipleHops"] = false

	    report(results["Description"])
	    return ar(results)
	end
    end


    local saddlePoint = maxs[1]
    local minimumEnergy1 = mins[1]

    local ESad = mep:energyAtPoint(saddlePoint)
    local E1 = mep:energyAtPoint(minimumEnergy1)

    --interactive("Pre")

    -- Working on rate:
    local hessian = mep:hessianAtPoint(saddlePoint)
    -- hessian = hessian:cutX(1,4):cutY(1,4) -- cut radial terms

    local radial_terms = {}
    for i=1,ns do
	radial_terms[i] = (i-1)*3+1
    end
    hessian = hessian:cutX(radial_terms):cutY(radial_terms) -- cut radial terms
    local evals, evecs = hessian:matEigen()
    local evalsSad = evals:copy() -- saddle point data
    local evecsSad = evecs:copy()


    evals = evals:toTable(1) -- convert to lua table

    -- find zero eigen value
    min_val, min_idx = evals[1], 1
    for s=2,ns*2 do
	if math.abs(evals[s]) < math.abs(min_val) then
	    min_val = evals[s]
	    min_idx = s
	end
    end

    table.remove(evals, min_idx) -- remove zero value
    
    local Lambda = makeDiagonal(evals)
    
    local V = evecs:cutY(min_idx) -- cut zero direction

    local M = {}
    local MG = 0
    for s=1,ns do
	M[s] = mep:spinInCoordinateSystem(1,s,"Canonical") -- r is first return value
	MG = MG + M[s] 
    end
    -- local M2 = mep:spinInCoordinateSystem(1,2,"Canonical") 
    -- local MG = M1 + M2

    
    function metricCC_at(idx)
	local p = {}
	for s=1,ns do
	    local _, _, ps = mep:spinInCoordinateSystem(idx, s, "Canonical")
	    p[s] = ps

	    if (1-ps^2) == 0 then
		if json.onNan then
		    json.onNan("This rotation has a fixed coordinate, p" .. s .. " = " .. ps .. " at saddle point")
		end
	    end
	end

	local diag = {}
	
	for s=1,ns do
	    table.insert(diag, (MG/M[s]) * (1-p[s]^2)^(-1))
	    table.insert(diag, (MG/M[s]) * (1-p[s]^2)^( 1))
	end
	
	return makeDiagonal(diag)
    end
    
    local metricCC = metricCC_at(saddlePoint)

    --[[
    local Gamma = Array.Double.new(4, 4, {
				       0,  MG/M1,     0,     0,
				  -MG/M1,      0,     0,     0,
				       0,      0,     0,     MG/M2,
				       0,      0,    -MG/M2, 0     })
    --]]
    local Gamma = Array.Double.new(ns*2, ns*2, 1)
    Gamma:zero()
    for s=1,ns do
        local x, y = (s-1)*2+1, (s-1)*2+1
        -- setting zeros are not needed but help reveal what is going on:
        Gamma:set({x,y  }, 0)        Gamma:set({x+1,y},   MG/M[s])
        Gamma:set({x,y+1}, -MG/M[s]) Gamma:set({x+1,y+1}, 0      )
    end


    local Tmatrix = Gamma:pairwiseScaleAdd(alpha, metricCC)

    local Tbar = V:matMul(Tmatrix):matMul(V:matTrans())

    local KArray = Lambda:matMul(Tbar):matEigen()
    local kappa =  KArray:min()/alpha

    mep:setCoordinateSystem("CanonicalX") --rotated coord. system
    hessian = mep:hessianAtPoint(minimumEnergy1)
    hessian = hessian:cutX(radial_terms):cutY(radial_terms)
    local n1 = hessian:matEigen()


    local denom = 2*math.pi*kB*T * Lambda:absoluted():matDet()
    
    local logS1 = (makeDiagonal(n1):matDet() / denom )^(1/2)

    local f12 = alpha / (1+alpha^2) * math.abs(gamma_cgs * kappa / MG) * logS1

    if f12 ~= f12 then -- it's nan
	if json.onNan then
	    f12 = json.onNan("f12 is nan")
	end
    end


    results["EnergyBarrier"] = ESad - E1
    results["AttemptFrequency"] = f12
    results["Event"] = make_event_function(sites, final, report1, string.format( "flipping site %s to %s", txtSites, txtDestination))
    results["InitialEnergy"] = mep:energyAtPoint(1)
    results["MaximumEnergy"] = mep:energyAtPoint(saddlePoint)
    results["FinalEnergy"] = mep:energyAtPoint(mep:numberOfPoints())
    results["Description"] = string.format("Physics for %s calculated" , txtSites)
    results["AtSingleWell"] = false
    results["ToSingleWell"] = false
    results["MultipleHops"] = false


    report(string.format("Calculating physics for %s: EB = %e ergs f0 = %e Hz", txtSites, results["EnergyBarrier"], results["AttemptFrequency"]))
    return ar(results)
end

--local MaxDegreesDifference = 5 -- to decide on unique sites
--local LowEnergy = 5e-13 * erg -- for clusters, auto-transitions

function GeneratePathDynamicsGraph(all_paths, MaxDegreesDifference, LowEnergy)
    if all_paths == nil then
	error("First argument must be a table of return values from PathDynamics()")
    end
    if MaxDegreesDifference == nil then
	error("Second argument must provide the Maximum number of average degrees between vector groups to decide equality (try 5)")
    end

    LowEnergy = LowEnergy or 0

    -- we will look at the InitialConfigurations to figure out how many start points we have
    local InitialConfiguration = {}

    local function totalAngularDifference(a,b)
	local sum, n = 0, 0
	for k,v in pairs(a) do
	    sum = sum + math.angleBetween(a[k], b[k])
	    n=n+1
	end
	return sum, n
    end

    local function averageAngularDifference(a,b)
	local sum,n = totalAngularDifference(a,b)
	return sum/n
    end

    local function sameConfiguration(a,b)
	local diff = averageAngularDifference(a,b)
	return diff <  MaxDegreesDifference * math.pi/180
    end

    -- working out numer of minima
    for k,v in pairs(all_paths) do
	local same_as = nil
	for test_idx,ic in pairs(InitialConfiguration) do
	    if sameConfiguration(ic,v.InitialConfiguration) then
		same_as = test_idx
	    end
	end

	if same_as == nil then
	    table.insert(InitialConfiguration, v.InitialConfiguration)
	end
    end

    local number_of_minima = table.maxn(InitialConfiguration)

    -- now we will create our graph. It will have nodes and (outgoing) edges
    local nodes = {}

    for i=1,number_of_minima do
	nodes[i] = {
	    Configuration=InitialConfiguration[i],
	    Energy=0, -- need to populate
	    Edges = {}, -- to be populated
	    Low_Energy_Partner = {}, 
	    High_Energy_Partner = {}, -- a partner who is not considered Low Energy
	    Cluster_Partner = {},  -- collection of nodes who are all mutual Low E Partners
	    Partner = {} -- High + Low
	}
    end

    local function which_node(cfg) -- find node with this cfg
	for k,n in pairs(nodes) do
	    if sameConfiguration(n.Configuration, cfg) then
		return k
	    end
	end
	-- interactive("Failed to find `cfg'")
    end

    -- now we will iterate through all_paths and assign them to nodes
    for k,p in pairs(all_paths) do
	local n = which_node(p.InitialConfiguration)
	if n then
	    table.insert(nodes[n].Edges, p)
	    nodes[n].Energy = p.InitialEnergy
	end
    end

    -- identifying low-energy and high-energy transitions (only single hops)
    local low_energy_edge, high_energy_edge = {}, {}
    for k,p in pairs(all_paths) do
	if p.MultipleHops == false then 
	    local a = which_node(p.InitialConfiguration)
	    local b = which_node(p.FinalConfiguration)
	    if p.EnergyBarrier <= LowEnergy then
		table.insert(low_energy_edge, {a,b})
	    else
		table.insert(high_energy_edge, {a,b})
	    end
	end
    end

    local function low_energy_edge_exists(ab)
	for k,v in pairs(low_energy_edge) do
	    if v[1] == ab[1] and v[2] == ab[2] then
		return true
	    end
	end
	return false
    end

    local function tableContains(t, test)
	for k,v in pairs(t) do
	    if v == test then
		return true
	    end
	end
	return false
    end

    -- now tell the nodes about their low-energy transitions
    -- and cluster partners
    for k,v in pairs(low_energy_edge) do
	local a,b = v[1], v[2]
	table.insert(nodes[a].Low_Energy_Partner, b)

	if low_energy_edge_exists({b,a}) then
	    table.insert(nodes[a].Cluster_Partner, b)
	end
    end
    -- and high energy edges
    for k,v in pairs(high_energy_edge) do
	local a,b = v[1], v[2]
	table.insert(nodes[a].High_Energy_Partner, b)
    end

    -- cluster partnership is transitive. 
    -- Tell clusters about all their partner's clusters until there 
    -- are no more to add. 
    -- (This will add the self reference, which is a good thing)
    local added_cluster_info = true
    local cluster_iteration = 0
    while added_cluster_info do
	cluster_iteration = cluster_iteration + 1
	if cluster_iteration > 100 then
        -- doing full error reporting:
	    local tmp = os.pid()
	    local cp_file = "cluster_problem_" .. tmp .. ".dat"
	    local er_file = "ERROR_" .. tmp .. ".txt"
	    local f = io.open(er_file, "w")
	    f:write("There was a problem combining clusters. The iteraction counter has exceeded 100, which is " ..
		    "unreasonable. We are saving the input to this function into `" .. cp_file .. "'. Perhaps " ..
		    "Jason can help.")
	    f:close()
	    checkpointSave(cp_file, all_paths, MaxDegreesDifference, LowEnergy)
	    error("Cluster problem. See file `"..er_file.."' for details.")
	    -- interactive("cluster_iteration > 100")
	end

	added_cluster_info = false
	for k,n in pairs(nodes) do
	    for i,cp in pairs(n.Cluster_Partner) do
		-- if my cluster partner's clusters partners list
		-- doesn't have me or my cluster partners then add them
		
		-- do you have me?
		if not tableContains(nodes[cp].Cluster_Partner, k) then
		    added_cluster_info = true
		    table.insert(nodes[cp].Cluster_Partner, k)
		end

		-- do you have my partners?
		for j,cp2 in pairs(n.Cluster_Partner) do
		    if not tableContains(nodes[cp].Cluster_Partner, cp2) then
			added_cluster_info = true
			table.insert(nodes[cp].Cluster_Partner, cp2)
		    end
		end
	    end
	end
    end

    -- finally we will combine Low and High energy into a single list for convenience
    for k1,n in pairs(nodes) do
	for k2,v in pairs(n.Low_Energy_Partner) do
	    table.insert(n.Partner, v)
	end
	for k2,v in pairs(n.High_Energy_Partner) do
	    table.insert(n.Partner, v)
	end
    end

    -- now we will build the convenience search function that
    -- takes 2 node indexes and will find the appropriate all_path element
    local function buildSearch()
	-- forming closures
        local _nodes = nodes 
        local sc = sameConfiguration
        local _all_paths = all_paths
	

        return function(a,b)
            local startCfg = _nodes[a].Configuration
            local endCfg   = _nodes[b].Configuration

            for k,p in pairs(_all_paths) do
		if sc(startCfg, p.InitialConfiguration) then
		    if sc(endCfg, p.FinalConfiguration) then
			return p
		    end
		end
	    end
	end
    end

    return nodes, buildSearch()
end


local function refine_mins(mep, mins)
    local _mins, _maxs = mep:criticalPoints()
    mins = mins or _mins
    for k,v in pairs(mins) do
	-- never ever, ever refine the start and end points:
	if v ~= 1 and v ~= np then
	    local iterations = 0
	    local ratio = 1 --mep:expensiveEnergyMinimizationAtPoint(v)
	    local h = 1e-8
	    while ratio > 1/50 and iterations < 20 do 
		-- we will continue to minimize until we hit the bottom
		ratio,_,_,h = mep:expensiveEnergyMinimizationAtPoint(v, 50, h)
		iterations = iterations + 1
	    end
	end
    end
end

local function refine_maxs(mep, maxs)
    local _mins, _maxs = mep:criticalPoints()
    maxs = maxs or _maxs
    for k,v in pairs(maxs) do
	-- never ever, ever refine the start and end points:
	if v ~= 1 and v ~= np then
	    local iterations = 0
	    local ratio = 1 --mep:expensiveGradientMinimizationAtPoint(v)
	    local h = 1e-8
	    while ratio > 1/50 and iterations < 20 do 
		-- we will continue to minimize until we hit the bottom
		ratio,_,_,h = mep:expensiveGradientMinimizationAtPoint(v, 50, h)
		iterations = iterations + 1
	    end
	end
    end
end

local function attempt_simplify(mep, equal_degrees)
    -- we'll look for multi-hops and loops
    -- first we'll push any mid-mins down to their local well
    -- and maxs to their minimum gradient
    -- goal is to have points line up so we can identify similar points
    local np = mep:numberOfPoints()
    mep:reduceToCriticalPoints()

    local mins, maxs = mep:criticalPoints()

    refine_mins(mep, mins)
    refine_maxs(mep, maxs)

    -- and then run the simplify algorithm to cut out internal loops
    mep:simplifyPath(equal_degrees * math.pi / 180)
    --local cut_points = np - mep:numberOfPoints()

    --mep:resamplePath(np) -- and re-expand the path
    --return cut_points
end


function AllPathDynamics(mep, minima, JSON)
    JSON = JSON or {}
    local function blank() end
    local report = JSON.Report or blank
    local PathDynamicsOnNan = JSON.PathDynamicsOnNan or error
    local PathDynamicsReport = JSON.PathDynamicsReport or blank
    local PathDynamicsOnMultipleHops = JSON.PathDynamicsOnMultipleHops or blank
    local SimplifyDegrees = JSON.SimplifyDegrees or 5
    local MultipleHopRetries = JSON.MultipleHopRetries or 2
    local all_paths = {}
    local number_of_minima = table.maxn(minima)
    local symmetricLandscape = JSON.SymmetricLandscape or false
    --local PathDynamicsRotationAxis = JSON.PathDynamicsRotationAxis or {0,0,1}
    --local PathDynamicsRotationRadians = JSON.PathDynamicsRotationAngle or 0
    local AllPathDynamicsMEP = JSON.AllPathDynamicsMEP or function() error("AllPathDynamicsMEP() is not optional") end

    for i=1,number_of_minima-1 do
        for j=i+1,number_of_minima do
	    local mep2 = mep:copy()

	    report("Creating path between minimum points " .. i .. " and " .. j)
            AllPathDynamicsMEP(mep, minima[i], minima[j], report)


            local pd_opts = {
                report=PathDynamicsReport, 
                onNan=PathDynamicsOnNan, 
                onMultipleHops=PathDynamicsOnMultipleHops
            }

	    report("Calculating physics from point " .. i .. " to " .. j)
	    local pd = PathDynamics(mep, pd_opts)
	    table.insert(all_paths, pd)
	    report("")

	    mep:reverse()
	    report("Calculating physics from point " .. j .. " to " .. i)
	    local pd = PathDynamics(mep, pd_opts)
	    table.insert(all_paths, pd)
	    report("")
	    
	end
    end

    return all_paths
end
