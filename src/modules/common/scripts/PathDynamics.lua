-- This file provides the function PathDynamics(mep) which, when given an MEP object containing only 1 maximum, will return a calculated description of the dynamics associated with the path in a table.  An optional second argument can be given to the function with a JSON style table. The following lists the keys that can be set for the JSON table:
--<dl>
--<dt>report</dt><dd>Function used to report internal progress of the algorithms, generally `print'</dd>
--<dt>onNan</dt> <dd>Function called when a nan is detected in the final stages of the calculation, generally `interactive' or `error'</dd>
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
--<dt>MEPComputePoints</dt><dd>Number of points to use in the internal mep:compute(), default 30</dd>
--<dt>MEPComputeSteps</dt><dd>Number of steps to use in the internal mep:compute(), default 40</dd>
--<dt>MEPComputeReport</dt><dd>Function to provide as the mep:compute report function, default `function() end'. The print function could be used.</dd>
--<dt>Report</dt><dd>Functon to use as the internal report function, default `function() end'. The print function could be used.</dd>
--<dt>PathDynamicsOnNan</dt><dd>Function to provide the PathDynamics function's OnNan option, default `error'.</dd>
--<dt>PathDynamicsReport</dt><dd>Function to provide the PathDynamics function's Report option, default `function() end'.</dd>
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
    local addResults = json.addResults or {}
    local report = json.report or function() end
    -- mpi root report functon
    local report1 = function(...) if mpi.get_rank() == 1 then report(...) end end
    local energyMinimizationSteps = json.EnergyMinimizationSteps or 100
    local gradientMinimizationSteps = json.GradientMinimizationSteps or 100

    local mep = mepOrig:copy()
    local sites = mep:sites()

    mep:setCoordinateSystem("Canonical")
    mep:setTolerance(0.0001)

    local function ar(res) -- add inputted results, add txt res
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


    local mins, maxs, all = mep:criticalPoints()
    -- record all minimum configurations
    for k,idx in pairs(mins) do
	results.MinimumConfigurations[k] = mep:point(idx)
    end 


    if table.maxn(maxs) > 1 then -- it's a double multi-hop transition. We don't deal with this
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


    -- working on individual points
    local mins, maxs, critPts = mep:criticalPoints()

    if maxs[1] then
	mep:expensiveGradientMinimizationAtPoint(maxs[1], gradientMinimizationSteps, 1e-7)
    end

    -- we should ahve good points coming in, no need to do this
    -- mep:expensiveEnergyMinimizationAtPoint(1, energyMinimizationSteps, 1e-7) -- 100 steps

    
    local txtDestination = txtPt(mep, mep:numberOfPoints(),"Cartesian")

    mep:reduceToPoints(critPts) -- reduce from N to 3 points
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
	    results["EnergyBarrier"] = 0
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

    -- Working on rate:
    local hessian = mep:hessianAtPoint(saddlePoint)
    hessian = hessian:cutX(1,4):cutY(1,4) -- cut radial terms
    local evals, evecs = hessian:matEigen()
    local evalsSad = evals:copy()
    local evecsSad = evecs:copy()


    evals = evals:toTable(1) -- convert to lua table

    -- find zero eigen value
    min_val, min_idx = evals[1], 1
    for i=2,4 do
	if math.abs(evals[i]) < math.abs(min_val) then
	    min_val = evals[i]
	    min_idx = i
	end
    end

    table.remove(evals, min_idx) -- remove zero value
    
    local Lambda = makeDiagonal(evals)
    
    local V = evecs:cutY(min_idx) -- cut zero direction

    local M1 = mep:spinInCoordinateSystem(1,1,"Canonical") -- r is first return value
    local M2 = mep:spinInCoordinateSystem(1,2,"Canonical") 

    local MG = M1 + M2

    
    function metricCC_at(idx)
	local _, _, p1 = mep:spinInCoordinateSystem(idx, 1, "Canonical")
	local _, _, p2 = mep:spinInCoordinateSystem(idx, 2, "Canonical")
	
	return makeDiagonal({
				(MG/M1) * (1-p1^2)^(-1),
				(MG/M1) * (1-p1^2)^( 1), 
				(MG/M2) * (1-p2^2)^(-1),
				(MG/M2) * (1-p2^2)^( 1) })
    end
    
    local metricCC = metricCC_at(saddlePoint)
    
    local Gamma = Array.Double.new(4, 4, {
				       0,  MG/M1,     0,     0,
				  -MG/M1,      0,     0,     0,
				       0,      0,     0,     MG/M2,
				       0,      0,    -MG/M2, 0     })

    local Tmatrix = Gamma:pairwiseScaleAdd(alpha, metricCC)

    local Tbar = V:matMul(Tmatrix):matMul(V:matTrans())

    local KArray = Lambda:matMul(Tbar):matEigen()
    local kappa =  KArray:min()/alpha

    mep:setCoordinateSystem("CanonicalX") --rotated coord. system
    hessian = mep:hessianAtPoint(minimumEnergy1)
    hessian = hessian:cutX(1,4):cutY(1,4) -- cut radius terms
    local n1 = hessian:matEigen()


    local denom = 2*math.pi*kB*T * Lambda:absoluted():matDet()
    
    local logS1 = (makeDiagonal(n1):matDet() / denom )^(1/2)

    local f12 = alpha / (1+alpha^2) * math.abs(gamma_cgs * kappa / MG) * logS1

    if f12 ~= f12 then -- it's nan
	if json.onNan then
	    json.onNan("f12 is nan")
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
    results["ToSingleWell"] = true
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

    LowEnergy = LowEnergy or nil

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
    while added_cluster_info do
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




function AllPathDynamics(mep, minima, JSON)
    JSON = JSON or {}
    local function blank() end
    local ComputeSteps = JSON.MEPComputeSteps or 40
    local ComputePoints = JSON.MEPComputePoints or 20
    local report = JSON.Report or blank
    local computeReport = JSON.MEPComputeReport or blank
    local PathDynamicsOnNan = JSON.PathDynamicsOnNan or error
    local PathDynamicsReport = JSON.PathDynamicsReport or blank

    local all_paths = {}
    local number_of_minima = table.maxn(minima)

    for i=1,number_of_minima-1 do
	for j=i+1,number_of_minima do
	    local mep2 = mep:copy()

	    mep2:setInitialPath({minima[i], minima[j]})
	    mep2:resamplePath(ComputePoints)
	    mep2:compute(ComputeSteps, {report=computeReport}) -- run MEP

	    report("Creating path between minimum points " .. i .. " and " .. j)
	    local pd = PathDynamics(mep2,  {report=PathDynamicsReport, onNan=PathDynamicsOnNan})
	    table.insert(all_paths, pd)
	    report("")

	    mep2:reverse()
	    report("Creating path between minimum points " .. j .. " and " .. i)
	    local pd = PathDynamics(mep2,  {report=PathDynamicsReport, onNan=PathDynamicsOnNan})
	    table.insert(all_paths, pd)
	    report("")
	    
	end
    end

    return all_paths
end

