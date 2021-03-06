-- KMC
local MODNAME = "KMC"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time
local methods = {} -- trying something new for style. Putting docs and code in a table which will later be used
-- to build metatable and help system. 

-- initialize/set/get internal data
-- "dd" means nothing, could read it as "do data"
local function dd(kmc, new_data)
    if new_data then
	kmc:_setinternaldata(new_data)
    end
    local t = kmc:_getinternaldata() or {}
    return t
end 

methods["setCustomData"] = {
    "Set custom data used in the implementation of the algorithm. It is common to use a table as data.",
    "1 Value: New custom data",
    "",
    function(kmc, value)
	local data = dd(kmc)
	data.user_data = value
	dd(kmc, data)
    end
}
methods["customData"] = {
    "Get custom data used in the implementation of the algorithm. It is common to use a table as data.",
    "",
    "1 Value: New custom data",
    function(kmc, value)
	local data = dd(kmc)
	return data.user_data
    end
}


methods["setMaxTimeStep"] = {
    "Set an upper limit on step size. If the fastest event is larger than this size then the time will be advanced without an event taking place. Default = 1e30",
    "1 Number: Maximum step size",
    "",
    function(kmc, value)
	local data = dd(kmc)
	data.max_step = value
	dd(kmc, data)
    end
}
methods["maxTimeStep"] = {
    "Get the upper limit on step size. If the fastest event is larger than this size then the time will be advanced without an event taking place",
    "",
    "1 Number: Maximum step size",
    function(kmc, value)
	local data = dd(kmc)
	return data.max_step or 1e30
    end
}


--[[
methods["setGroundStateEnergy"] = {
    "Set the energy of the ground state. This is used when looking at probabilities in some MC picking.",
    "1 Number: Ground state energy",
    "",
    function(kmc, value)
	local data = dd(kmc)
	data.ground_state_energy = value
	dd(kmc, data)
    end
}
methods["groundStateEnergy"] = {
    "Get the energy of the ground state. This is used when looking at probabilities in some MC picking.",
    "",
    "1 Number: Ground state energy",
    function(kmc)
	local data = dd(kmc)
	if data.ground_state_energy then
	    return data.ground_state_energy
	end
	error("Ground state energy not set, please use KMC:setGroundStateEnergy()")
    end
}
--]]

methods["setStartupFunction"] = {
    "Set a function that will be called before the loop over sites in the apply method. This will be called by all MPI processes.",
    "1 function: It is expected that the first argument is a KMC object and the second is the *SpinSystem* invovled in the calculation",
    "",
    function(kmc, sf)
	local data = dd(kmc)
	data.startup_function = sf
	dd(kmc, data)
    end
}
methods["startupFunction"] = {
    "Get the function that will be called before the loop over sites in the apply method.",
    "",
    "1 function: It is expected that the first argument is a KMC object and the second is the *SpinSystem* invovled in the calculation",
    function(kmc, sf)
	local data = dd(kmc)
	return (data.startup_function or function() end)
    end
}


methods["setProposedEventFunction"] = {
    "Set a function that will be called immediatelty before the selected event function. It will be passed the time and function of the fastest event as well as the position and the third return value from the eventFunction. The semantics of the third return value is opaque as far as the KMC framework is concerned. It is up to the user to decide what to do with it. A common choice is to have it be a table with information regarding the energy barrier of the event and clustering decisions to allow adaption of a cluster energy cutoff. It could also be used to record the sequence of events for a simulation. If this function is defined, it is responsible for retuning a boolean value. If it returns true then the event function will be called, otherwise it will not. If a proposedEvent function is not provided or it is set to nil then the selected event function will always be called.",
    "1 Function: The function will be passed a minimum time, the corresponding event function, the position of the event and the arbitrary 3rd return value from the create event function. It must return a boolean value which will determine if the KMC framework will call the event function on the SpinSystem.",
    "",
    function(kmc, sf)
	local data = dd(kmc)
	data.proposed_function = sf
	dd(kmc, data)
    end
}
methods["proposedEventFunction"] = {
    "Get the \"proposedEvent\" function. See KMC:setProposedEventFunction for details",
    "",
    "1 function: proposed event function.",
    function(kmc, sf)
	local data = dd(kmc)
	return (data.proposed_function or function() return true end)
    end
}



methods["setShutdownFunction"] = {
    "Set a function that will be called after the loop over sites in the apply method. This will be called by all MPI processes.",
    "1 function: It is expected that the first argument is a KMC object and the second is the *SpinSystem* invovled in the calculation and the third is a table of key-value pairs with keys of positions and values of event times. If you want all the event times you will need to combine the data manually from all the processes.",
    "",
    function(kmc, sf)
	local data = dd(kmc)
	data.shutdown_function = sf
	dd(kmc, data)
    end
}
methods["shutdownFunction"] = {
    "Get the function that will be called after the loop over sites in the apply method.",
    "",
    "1 function: It is expected that the first argument is a KMC object and the second is the *SpinSystem* invovled in the calculation and the third is a table of key-value pairs with keys of positions and values of event times.",
    function(kmc, sf)
	local data = dd(kmc)
	return (data.shutdown_function or function() end)
    end
}


methods["setEventFunction"] = {
    "Set the function used to compute site event time and action", 
    "1 Function: The function is expected to take a KMC object, a SpinSystem and a value representing a position. It is expected to return 1 number and 1 function. The number is the time associated with the event. The returned function should take a SpinSystem as an argument and, if called, do the event at the given site. If the function returns nil rather than 1 number and 1 function this signifies that there is no other energy minimum to transition to. This can happen, for example, when pinned by a large field.", 
    "",
    function(kmc, func)
	local data = dd(kmc)		
	if type(func_or_value) ~= type(function() end) then
	    local v = func_or_value
	    func_or_value = function() return v end
	end		
	data.EBAF = func
	dd(kmc, data)
    end
}
methods["eventFunction"] = {
    "Get the function used to compute the Energy Barrier and Attempt Frequency at a site.",
    "", 
    methods["setEventFunction"][3],
    function(kmc, func)
	local data = dd(kmc)
	return (data.EBAF or function() error("event function not set") end)
    end
}

methods["setPositions"] = {
    "Set a list of custom sites or values that the internal loop will iterate over. If this is not set (or it is set to nil) the default behaviour is to iterator over all sites in the SpinSystem.",
    "1 Table of values or nil: If nil then the default behaviour will be used otherwise the inner loop will iterate over all values in the 1D table. Each value will be provided as the position for the energy barrier/attempt frequency funtion",
    "",
    function(kmc, values)
	local data = dd(kmc)
	data.positions = values
	dd(kmc, data)
    end
}
methods["positions"] = {
    "Get the list of custom sites or values that the internal loop will iterate over. If this is not set then default behaviour is to iterator over all sites in the SpinSystem.",
    "",
    "1 Table of values or nil: If nil then the default behaviour will be used otherwise the inner loop will iterate over all values in the 1D return table. Each value will be provided as the position for the energy barrier/attempt frequency funtion. Note: The table returned is not a copy of the internal table so any changes to it will modify the internal behviour of the framework.",
    function(kmc)
	local data = dd(kmc)
	return data.positions or nil
    end
}


methods["setFields"] = {
    "Set the fields involved in the calculation",
    "1 Table of MagLua field operators. If a thermal operator is included you may set the temperature by interacting with it, otherwise use the setTemperature method.",
    "",
    function(kmc, fields)
	local data = dd(kmc)
	-- we will store 2 lists, the 1st is the unaltered input, the 2nd is the parsed by type
	local raw_fields = {}
	for k,v in pairs(fields) do
	    table.insert(raw_fields, v)
	end
	data.raw_fields = raw_fields
	data.fields_no_temp = {}
	data.fields_only_temp = {}

	for k,op in pairs(raw_fields) do
	    local temp_op = false
	    local mt = (getmetatable(op) or {})
	    if mt["slotName"] ~= nil then -- it's a MagLua operator
		if mt["temperature"] then -- it's probably a thermal operator
		    if op:slotName() == "Thermal" then -- let's call it a thermal operator
			temp_op = true
		    end
		end
	    end
	    if temp_op then
		table.insert(data.fields_only_temp, op)
	    else
		table.insert(data.fields_no_temp, op)
	    end
	end

	dd(kmc, data)
    end
}
methods["fields"] = {
    "Get the fields involved in the calculation",
    "1 Optional String or an Optional table of strings: If strings are provided they will be used as tests. Field operators with slot names that match one of the given strings will be added to the first return table, the remainder will be added to the second return table. If no strings are given then all fields will be added to the first table and none will be in the second.",
    "2 Tables of MagLua field operators:",
    function(kmc, test)
	local data = dd(kmc)
	if test == nil then
	    return (data.raw_fields  or {}), {}
	end
	if type(test) == type("a") then
	    test = {test}
	end

	local r1, r2 = {}, {}
	local d = (data.raw_fields  or {})
	for k,fop in pairs(d) do
	    local ok = false
	    for _,name in pairs(test) do
		if fop:slotName() == name then
		    ok = true
		end
	    end
	    if ok then
		table.insert(r1, fop)
	    else
		table.insert(r2, fop)
	    end
	end
	return r1, r2
    end
}

methods["setTemperature"] = {
    "Set the temperature used in the calculations. If a non-nil value is supplied it will override the temperature set in any provided thermal object. If a nil value is supplied (or called without an argument) the override will be broken and the temperature will be looked up from the thermal object",
    "1 Number or nil or nothing: The new temperature",
    "",
    function(kmc, temp)
	local data = dd(kmc)
	data.custom_temp = temp
	dd(kmc, data)
    end
}
methods["temperature"] = {
    "Get the temperature used in the calculations. If one has not been set with the setTemperature method and a thermal operator is present in the fields it's temperature will be returned. If neither of these cases are true then nil will be returned",
    "",
    "1 Number or nil: Temperature",
    function(kmc)
	local data = dd(kmc)
	if data.custom_temp then
	    return data.custom_temp
	end
	
	local tt = data.fields_only_temp or {}
	if tt[1] then
	    return tt[1]:temperature()
	end

	return nil
    end
}

methods["setRNG"] = {
    "Set the internal RNG (Random Number Generator). If an RNG is not set one will be created when required",
    "1 RNG or nil: New RNG. If the argument is nil then the internal RNG will be cleared and a new one will be required when required",
    "",
    function(kmc,rng)
	local data = dd(kmc)
	data.rng = rng
	dd(kmc,data)
    end
}
methods["RNG"] = {
    "Get the internal RNG (Random Number Generator).",
    "",
    "1 RNG: The internal RNG",
    function(kmc,rng)
	local data = dd(kmc)
	if data.rng == nil then
	    data.rng = Random.Isaac.new()
	end
	dd(kmc,data)
	return data.rng
    end
}


methods["setMPIComm"] = {
    "Set a custom MPI Communicator to be used in the apply method. If no custom communicator is given then the MPI_COMM_WORLD will be used. If nil is given then MPI_COMM_WORLD will be used",
    "1 MPI Communicator: Work group to be used when running the apply method",
    "",
    function(kmc,comm)
	local data = dd(kmc)
	data.comm = comm
	dd(kmc,data)
    end
}
methods["MPIComm"] = {
    "Get the custom MPI Communicator to be used in the apply method. If no custom communicator has been set then then the MPI_COMM_WORLD is returned",
    "",
    "1 MPI Communicator: Work group to be used when running the apply method",
    function(kmc,comm)
	local data = dd(kmc)
	return (data.comm or mpi.comm_world)
    end
}

-- function to pick smallest values with ties decided randomly
-- nil is considered the biggest of all numbers
local function index_of_smallest_value(values, rng)
    local low_index = {}

    for i,v in pairs(values) do
	if low_index[1] == nil then -- if we have nothing we'll take anything
	    low_index = {i}
	else
	    if values[low_index[1]] == nil then
		if values[i] == nil then -- tie with nils
		    table.insert(low_index, i)
		else -- anything beats nils
		    low_index = {i}
		end
	    else
		if values[i] < values[low_index[1]] then
		    low_index = {i}
		end
		if values[i] == values[low_index[1]] then
		    table.insert(low_index, i)
		end
	    end
	end
    end

    local n = table.maxn(low_index)
    if n < 2 then
	return low_index[1] -- could be a number, could be nil
    end

    local r = math.ceil(rng:uniform() * n)
    return low_index[r]
end

methods["extraReturnFunction"] = {
    "get the function that is called with arguments (min_time, min_action, min_pos, min_opaque) and whose return values are appended to the standard return value of :apply(ss)",
    "",
    "1 Function: Function to call to augment the standard return value of :apply(ss)",
    function(kmc)
        local data = dd(kmc)
        if data.extraReturnFunction == nil then
            data.extraReturnFunction = function() end
        end
        return data.extraReturnFunction
    end
}

methods["setExtraReturnFunction"] = {
    "Set the function that is called with arguments (min_time, min_action, min_pos, min_opaque) and whose return values are appended to the standard return value of :apply(ss)",
    "1 Function: Function to call to augment the standard return value of :apply(ss)",
    "",
    function(kmc, func)
        local data = dd(kmc)
        data.extraReturnFunction = func
    end
}


methods["apply"] = {
    "Take a single step of the KMC algorithm by computing and stepping to the next nearest event",
    "1 *SpinSystem*: SpinSystem to operator on.",
    "1 Boolean, Optional Values: Value indicating if a step was made or if time was added to the SpinSystem. The proposedEventFunction can disallow an event.",
    function(kmc, ss)
        local data = dd(kmc)
	local comm = kmc:MPIComm()
	local rng = kmc:RNG()
	local max_step = kmc:maxTimeStep()

	local event_f = kmc:eventFunction() -- energy barrier, attempt frequenct
	local start_f = kmc:startupFunction()
	local end_f = kmc:shutdownFunction()
	local proposed_f = kmc:proposedEventFunction()

	-- local det_fields  = data.fields_no_temp
	-- local temp_fields =  data.fields_only_temp

	if comm:get_rank() == nil then
	    return false-- this process isn't part of this computation
	end

	-- if det_fields == nil then
	--    error("Fields are not set. Use the :setFields method.")
	-- end

	start_f(kmc, ss)


	local all_times = {}
	local all_actions = {}
	local all_opaque = {} -- opaque user data 
	local all_pos = {}
	local local_idx = 1

	local function inner_loop(pos)
	    local time, action, opaque = event_f(kmc, ss, pos) -- energy barrier/attempt frequency

	    all_times[ local_idx ] = time
	    all_actions[ local_idx ] = action
	    all_pos[ local_idx ] = pos
	    all_opaque[ local_idx ] = opaque

	    local_idx = local_idx + 1
	end

	local ppp = kmc:positions()
	if ppp == nil then
	    local n = ss:nx() * ss:ny() * ss:nz()
	    for i in comm:range(1,n) do
		inner_loop( ss:indexToPosition(i) )
	    end
	else
	    local n = table.maxn(ppp)
	    for i in comm:range(1,n) do
		inner_loop( ppp[i] )
	    end
	end

	local min_idx = index_of_smallest_value(all_times, rng)

	local mta
	-- send all proposed time/events to root
	if min_idx then
	    mta = comm:gather(1, {all_times[min_idx], all_actions[min_idx], all_pos[min_idx], all_opaque[min_idx]}) 
	else
	    mta = comm:gather(1, {nil, nil, nil, nil})
	end

	-- only root will look for smallest as 
	-- there is randomness for ties
	if comm:get_rank() == 1 then 
	    all_times = {}
            all_actions = {}
            all_pos = {}
	    all_opaque = {}
	    for i=1,table.maxn(mta) do
		all_times[i]   = mta[i][1]
		all_actions[i] = mta[i][2]
		all_pos[i]     = mta[i][3]
		all_opaque[i]  = mta[i][4]
	    end

	    min_idx = index_of_smallest_value(all_times, rng)

	    if min_idx then
		mta = comm:bcast(1,  {all_times[min_idx], all_actions[min_idx], all_pos[min_idx], all_opaque[min_idx]})
	    else
		mta = comm:bcast(1,  {nil, nil, nil, nil})
	    end
	else
	    mta = comm:bcast(1, nil)
	end

	--  min time
	local min_time = mta[1]
	local min_action = mta[2]
	local min_pos = mta[3]
	local min_opaque = mta[4]
        local made_step = true

	if proposed_f(min_time, min_action, min_pos, min_opaque) then
	    if min_time then
		if min_time > max_step then
		    ss:setTime(ss:time() + max_step)
		else
		    -- do action, update time
		    min_action(ss)
		    ss:setTime(ss:time() + min_time)
		end
	    else
		-- we haven't found a transition (all single wells), 
		-- let's step forward without doing anything
		ss:setTime(ss:time() + max_step)
	    end
        else
            made_step = false
	end

	end_f(kmc, ss, all_times) -- do something custom
	return made_step, kmc:extraReturnFunction()(min_time, min_action, min_pos, min_opaque)
        
    end
}



-- inject above into existing metatable for KMC operator
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
		  if x == nil then
		      return help()
		  end
		  return help(x)
	      end

