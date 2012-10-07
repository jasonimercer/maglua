-- 
-- This support script implements the following integration types
-- 
-- RK1, RK2, RK3, RK4, RK4_38, RK6, K3, BS3


-- This function returns a function that implements the
-- Runge Kutta method based on the input Butcher Table
-- If a temp operator is given it will be applied after the integration step
function make_rk_step_function(ss, type, calcFieldFunc, llg, optional_temp)
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
	for i=2,nstep do
		ss_k[i] = ss:copy() -- make copies for RK
	end
	local function istep(ss_src, factors, ss_dest) -- apply a BT row
		llgOp:apply(ss_src, factors[1], ss_src, ss_dest)
		for i=2,nstep do
			if factors[i] and factors[i] ~= 0 then
				llgOp:apply(ss_dest, factors[i], ss_k[i], ss_dest)
			end
		end
	end
	local function sfunc(ss_input)
		ss_k[1] = ss_input or ss -- can operate on default ss or a given system
		cff(ss) -- calc field for base system
		for i = 1,nstep-1 do
			istep(ss, butcher[i], ss_k[i+1])
			cff(ss_k[i+1])
		end
		istep(ss, butcher[nstep], ss)
		
		if temp then
			ss:resetFields()
			temp:apply(ss)
			ss:sumFields()
			llgOp:apply(ss, false) --false = don't advance timestep
		end
	end
	return sfunc
end

