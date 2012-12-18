-- This support script implements the following integration types
-- 
-- <pre>RK1, RK2, RK3, RK4, RK4_38, RK6, K3, BS3</pre>
--
-- The integration types are available via the "make_rk_step_function(ss, type, fieldFunc, llg, optional_temp)" function which takes a *SpinSystem*, a type which is a string equal to one of the listed types above, a field function that takes a *SpinSystem* and computes the deterministic effective field, an LLG operator and an optional thermal operator. 
-- Example:
-- <pre>
-- dofile("maglua://RungeKutta.lua")
-- 
-- ss   = SpinSystem.new(40,40)
-- ex   = Exchange.new(ss)
-- dip  = Dipole.new(ss)
-- ani  = Anisotropy.new(ss)
-- temp = Thermal.new(ss, Random.Isaac.new())
-- llg  = LLG.Quaternion.new()
--
-- --1D interpolation object used as {time, temperature}
-- temperature = Interpolate.new({{0,20}, {20,2}, {198,0}, {1e8,0}})
--
-- max_time = 200
-- ss:setTimeStep(5e-2)
-- ex_str = 1/2
-- ani_str = 5/2
--
-- for i=1,ss:nx() do
-- 	for j=1,ss:ny() do
-- 		ss:setSpin({i,j}, {1,0,0})
-- 
-- 		ex:add({i,j}, {i+1,j}, ex_str)
-- 		ex:add({i,j}, {i-1,j}, ex_str)
-- 		ex:add({i,j}, {i,j+1}, ex_str)
-- 		ex:add({i,j}, {i,j-1}, ex_str)
-- 
-- 		ani:add({i,j}, {0,0,1}, ani_str)
-- 	end
-- end
-- 
-- function calcField(ss)
-- 	ss:resetFields()
-- 	ex:apply(ss)
-- 	dip:apply(ss)
-- 	ani:apply(ss)
-- 	ss:sumFields()
-- end
-- 
-- step = make_rk_step_function(ss, "RK4", calcField, llg, temp)
-- 
-- while ss:time() < max_time do
-- 	temp:set(temperature:value(ss:time()))
-- 	step()
-- end
-- </pre>

--function make_rk_step_function(ss, type, calcFieldFunc, dynamics_, llg, optional_temp)
function make_rk_step_function(a1,a2,a3,a4,a5,a6)
	-- working on making args adaptive
	local a = {a1,a2,a3,a4,a5,a6}
	local args = {}

	args["userdata"] = {}
	args["function"] = {}
	args["string"] = {}

	for k,v in pairs(a) do
		local t = type(v)
		args[t] = args[t] or {}
		table.insert(args[t], v)
	end

	local ss = args["userdata"][1]
	local calcFieldFunc = args["function"][1]
	local dynamics_ = args["function"][2]
	local llg = args["userdata"][2]
	local optional_temp = args["userdata"][2]
	local type = args["string"][1]	


	-- Butcher Table, the c_{i} column isn't included
	local butcher = {}

	butcher["RK2"] = {
		{1/2},
		{  0, 1},
	}
	butcher["RK2"].name = "Runge Kutta 2nd Order"


	butcher["RK1"] = {
		{1},
	}
	butcher["RK1"].name = "Euler"


	butcher["K3"] = {
		{1/2},
		{ -1,   2},
		{1/6,  2/3, 1/6}
	}
	butcher["K3"].name = "Kutta's 3rd Order"


	butcher["RK4"] = {
		{1/2},
		{   0, 1/2},
		{   0,    0,  1},
		{ 1/6,  1/3, 1/3, 1/6}
	}
	butcher["RK4"].name = "Runge Kutta 4th Order"


	butcher["RK3"] = {
		{ 1/3},
		{   0,  2/3},
		{ 1/4,    0,  3/4}
	}
	butcher["RK3"].name = "Runge Kutta 3rd Order"


	butcher["RK4_38"] = {
		{  1/3},
		{ -1/3,   1},
		{    1,  -1,   1},
		{  1/8, 3/8, 3/8, 1/8}
	}
	butcher["RK4_38"].name = "Runge Kutta 4th Order 3/8 Variant"


	butcher["RK6"] = {
		{    1/3},
		{      0,   2/3},
		{   1/12,  4/12,  -1/12},
		{  -1/16, 18/16,  -3/16,  -6/16},
		{      0,   9/8,   -3/8,   -6/8,    4/8},
		{   9/44,-36/44,  63/44,  72/44,      0,  -64/44},
		{ 11/120,     0, 81/120, 81/120,-32/120, -32/120, 11/120}
	}
	butcher["RK6"].name = "Runge Kutta 6th Order"


	butcher["BS3"] = {
		{ 1/2},
		{   0,  3/4},
		{ 2/9,  1/3,  4/9},
		{7/24,  1/4,  1/3,  1/8}
	}
	butcher["BS3"].name = "Bogacki-Shampine 3rd Order"

	if butcher[type] == nil then
		local tt = {}
		for k,v in pairs(butcher) do
			table.insert(tt, k)
		end
		tt = table.concat(tt, "\", \"")
		return error("RK type must be one of \"" .. tt .. "\"")
	end
	
	butcher = butcher[type]
	-- These local variables will be closures of the returned function:
	-- They'll exist but there will be no way to access them externally
	local nstep = table.maxn(butcher)
	local ss_k = {}
	local cff = calcFieldFunc
	local temp = optional_temp
	local llgOp = llg
	local dynamics = dynamics_ or function() end
	ss_k[1] = ss
	for i=2,nstep do
		ss_k[i] = ss_k[1]:copy() -- make copies for RK
	end
	local function istep(ss_src, factors, ss_dest) -- apply a BT row
		llgOp:apply(ss_src, factors[1], ss_src, ss_dest)
		for i=2,nstep do
			if factors[i] and factors[i] ~= 0 then
				llgOp:apply(ss_dest, factors[i], ss_k[i], ss_dest)
			end
		end
	end
	local function sfunc(ss_input, skip_temperature)
		ss_k[1] = ss_input or ss -- can operate on default ss or a given system
		dynamics(ss_k[1])
		cff(ss_k[1]) -- calc field for base system
		for i = 1,nstep-1 do
			ss_k[i+1]:setTimeStep(ss_k[1]:timeStep())
			istep(ss_k[1], butcher[i], ss_k[i+1])
			dynamics(ss_k[i+1])
			cff(ss_k[i+1])
		end
		istep(ss_k[1], butcher[nstep], ss_k[1])
		
		if temp and not skip_temperature then
			ss_k[1]:resetFields()
			temp:apply(ss_k[1])
			ss_k[1]:sumFields()
			llgOp:apply(ss_k[1], false) --false = don't advance timestep
		end
	end
	return sfunc
end

